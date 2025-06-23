//----------------------------------------------------------------------------//
//                                                                            //
// ozz-animation is hosted at http://github.com/guillaumeblanc/ozz-animation  //
// and distributed under the MIT License (MIT).                               //
//                                                                            //
// Copyright (c) Guillaume Blanc                                              //
//                                                                            //
// Permission is hereby granted, free of charge, to any person obtaining a    //
// copy of this software and associated documentation files (the "Software"), //
// to deal in the Software without restriction, including without limitation  //
// the rights to use, copy, modify, merge, publish, distribute, sublicense,   //
// and/or sell copies of the Software, and to permit persons to whom the      //
// Software is furnished to do so, subject to the following conditions:       //
//                                                                            //
// The above copyright notice and this permission notice shall be included in //
// all copies or substantial portions of the Software.                        //
//                                                                            //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR //
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   //
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    //
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER //
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    //
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        //
// DEALINGS IN THE SOFTWARE.                                                  //
//                                                                            //
//----------------------------------------------------------------------------//

#include "ozz/animation/offline/motion_extractor.h"

#include <cassert>
#include <queue>

#include "ozz/animation/offline/animation_builder.h"
#include "ozz/animation/offline/raw_animation_utils.h"
#include "ozz/animation/offline/raw_animation.h"
#include "ozz/animation/offline/raw_track.h"
#include "ozz/animation/offline/raw_track_utils.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/local_to_model_job.h"
#include "ozz/animation/runtime/sampling_job.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/animation/runtime/skeleton_utils.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/base/maths/transform.h"

namespace ozz {
namespace animation {
namespace offline {

namespace {
ozz::math::Transform BuildReference(
    MotionExtractor::Reference _position_reference,
    MotionExtractor::Reference _rotation_reference,
    const ozz::math::Transform& _skeleton_ref,
    const RawAnimation::JointTrack& _track) {
  auto ref = ozz::math::Transform::identity();

  // Position reference
  switch (_position_reference) {
    case MotionExtractor::Reference::kSkeleton: {
      ref.translation = _skeleton_ref.translation;
    } break;
    case MotionExtractor::Reference::kAnimation: {
      if (!_track.translations.empty()) {
        ref.translation = _track.translations[0].value;
      }
    } break;
    default:
      break;
  }

  // Rotation reference
  switch (_rotation_reference) {
    case MotionExtractor::Reference::kSkeleton: {
      ref.rotation = _skeleton_ref.rotation;
    } break;
    case MotionExtractor::Reference::kAnimation: {
      if (!_track.rotations.empty()) {
        ref.rotation = _track.rotations[0].value;
      }
    } break;
    default:
      break;
  }
  return ref;
}

struct ProcessHeightArgs {
  int root_joint;
  MotionExtractor::Reference reference;
  float bottom_threshold;
};

bool ProcessHeight(const ProcessHeightArgs& _args, const Skeleton& _skeleton,
                   const RawAnimation& _animation_in,
                   RawAnimation* _animation_out,
                   ozz::vector<float>* _heights_out) {
  ozz::vector<ozz::math::Float4x4> models;
  models.resize(_skeleton.num_joints());
  ozz::vector<ozz::math::SoaTransform> locals;
  locals.resize(_skeleton.num_soa_joints());
  ozz::animation::SamplingJob::Context context;
  context.Resize(_skeleton.num_joints());

  float skeleton_bottom = 0;
  if (_args.reference == MotionExtractor::Reference::kSkeleton) {
    ozz::animation::LocalToModelJob stm_job;
    stm_job.skeleton = &_skeleton;
    stm_job.input = _skeleton.joint_rest_poses();
    stm_job.output = make_span(models);
    if (!stm_job.Run()) {
      return false;
    }
    skeleton_bottom = INFINITY;
    for each (ozz::math::Float4x4 mat in models) {
      skeleton_bottom = fminf(skeleton_bottom, mat.cols[3].m128_f32[1]);
    }
  }

  // Build temporary animation
  AnimationBuilder builder;
  unique_ptr<Animation> animation = builder(_animation_in);
  if (!animation) {
    return false;
  }

  // Collect time points (keep same with AnimationBuilder)
  std::vector<float> times;
  times.reserve(animation->timepoints().size());
  auto insert_time = [&times](float time) {
    auto it = std::lower_bound(times.begin(), times.end(), time);
    if (it == times.end() || *it != time) {
      times.insert(it, time);
    }
  };
  for each (auto track in _animation_in.tracks) {
    for each (auto t in track.translations) {
      insert_time(t.time);
    }
    for each (auto r in track.rotations) {
      insert_time(r.time);
    }
    for each (auto s in track.scales) {
      insert_time(s.time);
    }
  }

  _heights_out->reserve(times.size());
  RawAnimation::JointTrack::Translations translations;
  translations.reserve(times.size());

  for (size_t i = 0; i < times.size(); ++i) {
    float time = times[i];
    float ratio = (time - times.front()) / times.back();

    ozz::animation::SamplingJob sampling_job;
    sampling_job.animation = &*animation;
    sampling_job.context = &context;
    sampling_job.ratio = ratio;
    sampling_job.output = make_span(locals);
    if (!sampling_job.Run()) {
      return false;
    }

    ozz::animation::LocalToModelJob ltm_job;
    ltm_job.skeleton = &_skeleton;
    ltm_job.input = make_span(locals);
    ltm_job.output = make_span(models);
    if (!ltm_job.Run()) {
      return false;
    }

    float bottom = INFINITY;
    for each (auto mat in models) {
      bottom = fminf(bottom, mat.cols[3].m128_f32[1]);
    }
    if (i == 0 && _args.reference == MotionExtractor::Reference::kAnimation) {
      skeleton_bottom = bottom;
    }

    if (fabsf(bottom - skeleton_bottom) <= _args.bottom_threshold) {
      _heights_out->push_back(0.0);
    } else {
      _heights_out->push_back(bottom - skeleton_bottom);
    }

    RawAnimation::TranslationKey key;
    key.time = time;
    int root_joint = _args.root_joint;
    key.value = math::Float3(
        locals[root_joint / 4].translation.x.m128_f32[root_joint % 4],
        locals[root_joint / 4].translation.y.m128_f32[root_joint % 4],
        locals[root_joint / 4].translation.z.m128_f32[root_joint % 4]);
    translations.push_back(key);
  }

  *_animation_out = _animation_in;
  _animation_out->tracks[0].translations = translations;
  return true;
}
}  // namespace

bool MotionExtractor::operator()(const RawAnimation& _input,
                                 const Skeleton& _skeleton,
                                 RawFloat3Track* _motion_position,
                                 RawQuaternionTrack* _motion_rotation,
                                 RawAnimation* _output) const {
  // Cannot read/write from/to the same animation.
  if (&_input == _output) {
    return false;
  }

  // All outputs are expected to be valid.
  if (!_output || !_motion_position || !_motion_rotation) {
    return false;
  }

  // Animation must match skeleton.
  if (_input.num_tracks() != _skeleton.num_joints()) {
    return false;
  }

  // Root index must be within skeleton range.
  if (root_joint < 0 || root_joint >= _skeleton.num_joints()) {
    return false;
  }

  // Validate animation.
  if (!_input.Validate()) {
    return false;
  }

  // Extract root motion
  // -----------------------------------------------------------------------------

  ozz::vector<float> heights;
  if (position_settings.bottom) {
    ProcessHeightArgs args;
    args.root_joint = root_joint;
    args.reference = position_settings.reference;
    args.bottom_threshold = position_settings.bottom_threshold;
    if (!ProcessHeight(args, _skeleton, _input, _output, &heights)) {
      return false;
    }
  } else {
    *_output = _input; // Copy output animation
  }

  // Track to extract motion from
  const auto& input_track = _input.tracks[root_joint];
  auto& output_track = _output->tracks[root_joint];

  // Compute extraction reference
  auto ref =
      BuildReference(position_settings.reference, rotation_settings.reference,
                     GetJointLocalRestPose(_skeleton, root_joint), input_track);

  // Process root motion height first
  // -----------------------------------------------------------------------------

  // Copy function, used to copy aniamtion keyframes to motion keyframes.
  auto extract = [duration = _input.duration](const auto& _keframes,
                                              auto _extract, auto& output) {
    output.clear();
    for (const auto& joint_key : _keframes) {
      const auto& motion = _extract(joint_key.value);
      output.push_back({ozz::animation::offline::RawTrackInterpolation::kLinear,
                        joint_key.time / duration, motion});
    }
  };

  // Copies root position, selecting only expecting components.
  const math::Float3 position_mask{1.f * position_settings.x,
                                   1.f * position_settings.y,
                                   1.f * position_settings.z};
  extract(
      _output->tracks[root_joint].translations,
      [&mask = position_mask, &ref = ref.translation](const auto& _joint) {
        return (_joint - ref) * mask;
      },
      _motion_position->keyframes);
  if (position_settings.bottom) {
    for (size_t i = 0; i < heights.size(); ++i) {
      _motion_position->keyframes[i].value.y = heights[i];
    }
  }

  // Copies root rotation, selecting only expecting decomposed rotation
  // components.
  const math::Float3 rotation_mask{1.f * rotation_settings.y,   // Yaw
                                   1.f * rotation_settings.x,   // Pitch
                                   1.f * rotation_settings.z};  // Roll
  extract(
      input_track.rotations,
      [&mask = rotation_mask, &ref = ref.rotation](const auto& _joint) {
        const auto euler = ToEuler(_joint * Conjugate(ref));
        return math::Quaternion::FromEuler(euler * mask);
      },
      _motion_rotation->keyframes);

  // Bake
  // -----------------------------------------------------------------------------

  // Extract root motion rotation from the animation, aka bake it.
  if (rotation_settings.bake) {
    assert(output_track.rotations.size() == _motion_rotation->keyframes.size());
    for (size_t i = 0; i < output_track.rotations.size(); i++) {
      const auto& motion_q = _motion_rotation->keyframes[i].value;
      auto& joint_q = output_track.rotations[i].value;
      joint_q = Conjugate(motion_q) * joint_q;
    }
  }

  // Extract root motion position from the animation, aka bake it.
  if (position_settings.bake) {
    assert(output_track.translations.size() ==
           _motion_position->keyframes.size());
    for (size_t i = 0; i < output_track.translations.size(); i++) {
      const auto& motion_p = _motion_position->keyframes[i].value;
      auto& joint_p = output_track.translations[i].value;
      joint_p = joint_p - motion_p;
    }
  }

  // Loopify
  // -----------------------------------------------------------------------------
  // Distributes the difference between the first and last keyframes all along
  // animation duration, so tha animation can loop.
  auto loopify = [](auto& _keyframes, auto _diff, auto _lerp) {
    if (_keyframes.size() < 2) {
      return;
    }
    const auto delta = _diff(_keyframes.front().value, _keyframes.back().value);
    for (size_t i = 0; i < _keyframes.size(); i++) {
      const float alpha = i / (_keyframes.size() - 1.f);
      auto& value = _keyframes[i].value;
      value = _lerp(value, delta, alpha);
    }
  };

  // Loopify translations
  if (rotation_settings.loop) {
    loopify(
        _motion_rotation->keyframes,
        [](auto _front, auto _back) { return _front * Conjugate(_back); },
        [](auto _value, auto _delta, float _alpha) {
          return NLerp(math::Quaternion::identity(), _delta, _alpha) * _value;
        });
  }

  // Loopify rotations
  if (position_settings.loop) {
    loopify(
        _motion_position->keyframes,
        [](auto _front, auto _back) { return _front - _back; },
        [](auto _value, auto _delta, float _alpha) {
          return _delta * _alpha + _value;
        });
  }

  // Fixup animation translations.
  // -----------------------------------------------------------------------------
  // When root motion is applied, then root rotation is applied before joint
  // translation. Hence joint's translation should be corrected to support this
  // new composition order.
  if (rotation_settings.bake) {  // Considers that if rotation is baked, then
                                 // motion rotation will be applied
    for (size_t i = 0; i < output_track.translations.size(); i++) {
      const auto& motion_p_key = _motion_position->keyframes[i];
      auto& joint_p = output_track.translations[i].value;

      // Sample motion rotation (as it might not have the same number of
      // keyframes as translations)
      math::Quaternion motion_q;
      if (!SampleTrack(*_motion_rotation, motion_p_key.ratio, &motion_q)) {
        return false;
      }

      joint_p = TransformVector(Conjugate(motion_q), joint_p);
    }
  }

  // Validate outputs
  bool success = true;
  success &= _motion_position->Validate();
  success &= _motion_rotation->Validate();
  success &= _output->Validate();

  return success;
}
}  // namespace offline
}  // namespace animation
}  // namespace ozz
