/*
 * Copyright (c) 2018, XMotors.ai. All rights reserved.
 *
 */

// Add plugin for maskrcnn [caizw@20190905]
// REGISTER_BUILTIN_PLUGIN("GenerateProposalsOp", GenerateProposalsOpPlugin);
REGISTER_BUILTIN_PLUGIN("CollectAndDisOp", CollectAndDisOpPlugin);
REGISTER_BUILTIN_PLUGIN("RoIAlign", RoIAlignPlugin);
REGISTER_BUILTIN_PLUGIN("BatchPermutation", BatchPermutationPlugin);
REGISTER_BUILTIN_PLUGIN("GemvInt8", GemvInt8Plugin);
REGISTER_BUILTIN_PLUGIN("DecodeAndNMS", DecodeAndNMSPlugin);
