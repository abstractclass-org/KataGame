#ifndef NEURALNET_MODELVERSION_H_
#define NEURALNET_MODELVERSION_H_

// Model versions
namespace NNModelVersion {

  constexpr int latestModelVersionImplemented = 23;
  //Enclosure fork: every model version reads the game's inputs (V8, V9 in version 18, V10 in 19, V11 in 20, V12 in
  //21, V13 in 22, V14 in 23); Go's V3-V7 are not used.
  constexpr int latestInputsVersionImplemented = 14;
  //Self-play writes its training rows in this version's inputs (selfplay.cpp), whatever its network reads.
  constexpr int defaultModelVersion = 18;

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
