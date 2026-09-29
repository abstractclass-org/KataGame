#ifndef NEURALNET_MODELVERSION_H_
#define NEURALNET_MODELVERSION_H_

// Model versions
namespace NNModelVersion {

  constexpr int latestModelVersionImplemented = 17;
  //Enclosure fork: every model version reads the Enclosure inputs (V8); Go's V3-V7 are not used.
  constexpr int latestInputsVersionImplemented = 8;
  constexpr int defaultModelVersion = 17;

  constexpr int oldestModelVersionImplemented = 8;
  constexpr int oldestInputsVersionImplemented = 8;

  // Which V* feature version from NNInputs does a given model version consume?
  int getInputsVersion(int modelVersion);

  // Convenience functions, feeds forward the number of features and the size of
  // the row vector that the net takes as input
  int getNumSpatialFeatures(int modelVersion);
  int getNumGlobalFeatures(int modelVersion);

  // SGF metadata encoder input versions
  int getNumInputMetaChannels(int metaEncoderVersion);

}  // namespace NNModelVersion

#endif  // NEURALNET_MODELVERSION_H_
