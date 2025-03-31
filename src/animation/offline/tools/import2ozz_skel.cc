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

#include "animation/offline/tools/import2ozz_skel.h"

#include <cstdlib>
#include <cstring>
#include <iomanip>

#include "animation/offline/tools/import2ozz_config.h"

#include "ozz/animation/offline/tools/import2ozz.h"

#include "ozz/animation/offline/raw_skeleton.h"
#include "ozz/animation/offline/skeleton_builder.h"

#include "ozz/animation/runtime/skeleton.h"

#include "ozz/base/containers/map.h"
#include "ozz/base/containers/set.h"

#include "ozz/base/io/archive.h"
#include "ozz/base/io/stream.h"

#include "ozz/base/memory/unique_ptr.h"

#include "ozz/base/log.h"

#include <json/json.h>

namespace ozz {
namespace animation {
namespace offline {
namespace {

// Uses a set to detect names uniqueness.
typedef ozz::set<const char*, ozz::str_less> Names;

bool ValidateJointNamesUniquenessRecurse(
    const RawSkeleton::Joint::Children& _joints, Names* _names) {
  for (size_t i = 0; i < _joints.size(); ++i) {
    const RawSkeleton::Joint& joint = _joints[i];
    const char* name = joint.name.c_str();
    if (!_names->insert(name).second) {
      ozz::log::Err()
          << "Skeleton contains at least one non-unique joint name \"" << name
          << "\", which is not supported." << std::endl;
      return false;
    }
    if (!ValidateJointNamesUniquenessRecurse(_joints[i].children, _names)) {
      return false;
    }
  }
  return true;
}

bool ValidateJointNamesUniqueness(const RawSkeleton& _skeleton) {
  Names joint_names;
  return ValidateJointNamesUniquenessRecurse(_skeleton.roots, &joint_names);
}

void LogHierarchy(const RawSkeleton::Joint::Children& _children,
                  int _depth = 0) {
  const std::streamsize pres = ozz::log::LogV().stream().precision();
  for (size_t i = 0; i < _children.size(); ++i) {
    const RawSkeleton::Joint& joint = _children[i];
    ozz::log::LogV() << std::setw(_depth) << std::setfill('.') << "";
    ozz::log::LogV() << joint.name.c_str() << std::setprecision(4)
                     << " t: " << joint.transform.translation.x << ", "
                     << joint.transform.translation.y << ", "
                     << joint.transform.translation.z
                     << " r: " << joint.transform.rotation.x << ", "
                     << joint.transform.rotation.y << ", "
                     << joint.transform.rotation.z << ", "
                     << joint.transform.rotation.w
                     << " s: " << joint.transform.scale.x << ", "
                     << joint.transform.scale.y << ", "
                     << joint.transform.scale.z << std::endl;

    // Recurse
    LogHierarchy(joint.children, _depth + 1);
  }
  ozz::log::LogV() << std::setprecision(static_cast<int>(pres));
}

bool ClipSkeleton(const Json::Value js_obj,
                  const RawSkeleton::Joint::Children& _raw_children,
                  RawSkeleton::Joint::Children* _clipped_children,
                  RawSkeleton::Joint::Children* _renamed_children) {
  if (js_obj.isNull()) {
    return true;

  } else if (js_obj.isObject()) {
    const Json::Value& js_name = js_obj["name"];
    if (!js_name.isString()) {
      ozz::log::Err() << "Json mapping (bad name)" << std::endl;
      return false;
    }
    std::string name = js_name.asString();

    const Json::Value& js_rename = js_obj["rename"];
    std::string rename = js_rename.isString() ? js_rename.asString() : "";
    const bool skip = js_obj["skip"].asBool();
    const bool optional = js_obj["optional"].asBool();

    auto raw_it =
        std::find_if(_raw_children.begin(), _raw_children.end(),
                     [&](auto it) { return it.name == name.c_str(); });
    if (raw_it == _raw_children.end()) {
      if (optional) {
        return ClipSkeleton(js_obj["next"], _raw_children, _clipped_children,
                            _renamed_children);
      } else {
        ozz::log::Err() << "Json mapping (joint '" << name << "' not found)"
                        << std::endl;
        return false;
      }

    } else {
      if (skip) {
        return ClipSkeleton(js_obj["next"], raw_it->children, _clipped_children,
                            _renamed_children);
      } else {
        ozz::animation::offline::RawSkeleton::Joint clipped_joint;
        clipped_joint.name = raw_it->name;
        clipped_joint.transform = raw_it->transform;

        ozz::animation::offline::RawSkeleton::Joint renamed_joint;
        renamed_joint.name =
            rename.empty() ? raw_it->name : ozz::string(rename.c_str());
        renamed_joint.transform = raw_it->transform;

        if (ClipSkeleton(js_obj["next"], raw_it->children,
                         &clipped_joint.children, &renamed_joint.children)) {
          _clipped_children->push_back(clipped_joint);
          _renamed_children->push_back(renamed_joint);
          return true;
        } else {
          return false;
        }
      }
    }

  } else if (js_obj.isArray()) {
    for (auto js_it = js_obj.begin(); js_it != js_obj.end(); ++js_it) {
      if (!ClipSkeleton(*js_it, _raw_children, _clipped_children,
                        _renamed_children)) {
        return false;
      }
    }
    return true;

  } else {
    ozz::log::Err() << "Json mapping (next type)" << std::endl;
    return false;
  }
}
}  // namespace

bool ImportSkeleton(const Json::Value& _config, OzzImporter* _importer,
                    const ozz::Endianness _endianness) {
  const Json::Value& skeleton_config = _config["skeleton"];
  const Json::Value& import_config = skeleton_config["import"];

  // First check that we're actually expecting to import a skeleton.
  if (!import_config["enable"].asBool()) {
    ozz::log::Log() << "Skeleton build disabled, import will be skipped."
                    << std::endl;
    return true;
  }

  // Setup node types import properties.
  const Json::Value& types_config = import_config["types"];
  OzzImporter::NodeType types = {};
  types.skeleton = types_config["skeleton"].asBool();
  types.marker = types_config["marker"].asBool();
  types.camera = types_config["camera"].asBool();
  types.geometry = types_config["geometry"].asBool();
  types.light = types_config["light"].asBool();
  types.null = types_config["null"].asBool();
  types.any = types_config["any"].asBool();
  RawSkeleton raw_skeleton;
  if (!_importer->Import(&raw_skeleton, types)) {
    ozz::log::Err() << "Failed to import skeleton." << std::endl;
    return false;
  }

  // Log skeleton hierarchy
  if (ozz::log::GetLevel() == ozz::log::kVerbose) {
    LogHierarchy(raw_skeleton.roots);
  }

  // Non unique joint names are not supported.
  if (!(ValidateJointNamesUniqueness(raw_skeleton))) {
    // Log Err is done by the validation function.
    return false;
  }

  // Needs to be done before opening the output file, so that if it fails then
  // there's no invalid file outputted.
  unique_ptr<Skeleton> skeleton;
  if (!import_config["raw"].asBool()) {
    // Builds runtime skeleton.
    ozz::log::Log() << "Builds runtime skeleton." << std::endl;
    SkeletonBuilder builder;
    skeleton = builder(raw_skeleton);
    if (!skeleton) {
      ozz::log::Err() << "Failed to build runtime skeleton." << std::endl;
      return false;
    }
  }

  // Prepares output stream. File is a RAII so it will close automatically at
  // the end of this scope.
  // Once the file is opened, nothing should fail as it would leave an invalid
  // file on the disk.
  {
    const char* filename = skeleton_config["filename"].asCString();
    ozz::log::Log() << "Opens output file: " << filename << std::endl;
    ozz::io::File file(filename, "wb");
    if (!file.opened()) {
      ozz::log::Err() << "Failed to open output file: \"" << filename << "\"."
                      << std::endl;
      return false;
    }

    // Initializes output archive.
    ozz::io::OArchive archive(&file, _endianness);
    // Fills output archive with the skeleton.
    if (import_config["raw"].asBool()) {
      ozz::log::Log() << "Outputs RawSkeleton to binary archive." << std::endl;
      archive << raw_skeleton;
    } else {
      ozz::log::Log() << "Outputs Skeleton to binary archive." << std::endl;
      archive << *skeleton;
    }
    ozz::log::Log() << "Skeleton binary archive successfully outputted."
                    << std::endl;
  }

  return true;
}

const char* FILE_KEYS[2] = {"clipped_file", "renamed_file"};
const int CLIPPED = 0;
const int RENAMED = 1;
const int KEY_MAX = 2;

bool ImportClipSkeleton(const Json::Value& _config, const Json::Value& _mapping,
                    OzzImporter* _importer, const ozz::Endianness _endianness) {
  const Json::Value& skeleton_config = _config["skeleton"];
  const Json::Value& import_config = skeleton_config["import"];

  // First check that we're actually expecting to import a skeleton.
  if (!import_config["enable"].asBool()) {
    ozz::log::Log() << "Skeleton build disabled, import will be skipped."
                    << std::endl;
    return true;
  }

  // Setup node types import properties.
  const Json::Value& types_config = import_config["types"];
  OzzImporter::NodeType types = {};
  types.skeleton = types_config["skeleton"].asBool();
  types.marker = types_config["marker"].asBool();
  types.camera = types_config["camera"].asBool();
  types.geometry = types_config["geometry"].asBool();
  types.light = types_config["light"].asBool();
  types.null = types_config["null"].asBool();
  types.any = types_config["any"].asBool();

  RawSkeleton raw_skeleton;
  if (!_importer->Import(&raw_skeleton, types)) {
    ozz::log::Err() << "Failed to import skeleton." << std::endl;
    return false;
  }

  RawSkeleton raw_skeletons[2] = {};
  if (!ClipSkeleton(_mapping, raw_skeleton.roots,
                    &raw_skeletons[CLIPPED].roots,
                    &raw_skeletons[RENAMED].roots)) {
    // Log Err is done by the clip function.
    return false;
  }

  // Log skeleton hierarchy
  if (ozz::log::GetLevel() == ozz::log::kVerbose) {
    LogHierarchy(raw_skeletons[CLIPPED].roots);
  }

  // Non unique joint names are not supported.
  if (!(ValidateJointNamesUniqueness(raw_skeletons[CLIPPED])) ||
      !(ValidateJointNamesUniqueness(raw_skeletons[RENAMED]))) {
    // Log Err is done by the validation function.
    return false;
  }

  // Needs to be done before opening the output file, so that if it fails then
  // there's no invalid file outputted.
  unique_ptr<Skeleton> skeletons[2] = {};
  if (!import_config["raw"].asBool()) {
    // Builds runtime skeleton.
    ozz::log::Log() << "Builds runtime skeleton." << std::endl;
    for (int idx = 0; idx < KEY_MAX; ++idx) {
      SkeletonBuilder builder;
      skeletons[idx] = builder(raw_skeletons[idx]);
      if (!skeletons[idx]) {
        ozz::log::Err() << "Failed to build runtime skeleton." << std::endl;
        return false;
      }
    }
  }

  // Prepares output stream. File is a RAII so it will close automatically at
  // the end of this scope.
  // Once the file is opened, nothing should fail as it would leave an invalid
  // file on the disk.
  for (int idx = 0; idx < KEY_MAX; ++idx) {
    const char* filename = skeleton_config[FILE_KEYS[idx]].asCString();
    ozz::log::Log() << "Opens output file: " << filename << std::endl;
    ozz::io::File file(filename, "wb");
    if (!file.opened()) {
      ozz::log::Err() << "Failed to open output file: \"" << filename << "\"."
                      << std::endl;
      return nullptr;
    }

    // Initializes output archive.
    ozz::io::OArchive archive(&file, _endianness);

    // Fills output archive with the skeleton.
    if (import_config["raw"].asBool()) {
      ozz::log::Log() << "Outputs RawSkeleton to binary archive." << std::endl;
      archive << raw_skeletons[idx];
    } else {
      ozz::log::Log() << "Outputs Skeleton to binary archive." << std::endl;
      archive << *skeletons[idx];
    }
    ozz::log::Log() << "Skeleton binary archive successfully outputted."
                    << std::endl;
  }

  return true;
}
}  // namespace offline
}  // namespace animation
}  // namespace ozz
