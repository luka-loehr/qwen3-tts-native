def buildkit_slsa_v02:
  .SLSA.buildType == "https://mobyproject.org/buildkit@v1"
  and (.SLSA.invocation.parameters | type) == "object"
  and .SLSA.metadata.completeness.parameters == true
  and .SLSA.metadata.completeness.environment == true;

def buildkit_slsa_v1:
  .SLSA.buildDefinition.buildType
    == "https://github.com/moby/buildkit/blob/master/docs/attestations/slsa-definitions.md"
  and (.SLSA.buildDefinition.externalParameters | type) == "object"
  and (.SLSA.buildDefinition.externalParameters.configSource | type) == "object"
  and (.SLSA.buildDefinition.externalParameters.request | type) == "object"
  and (.SLSA.buildDefinition.internalParameters | type) == "object"
  and (.SLSA.buildDefinition.internalParameters.buildConfig | type) == "object"
  and (.SLSA.buildDefinition.internalParameters.builderPlatform | type) == "string"
  and (.SLSA.buildDefinition.internalParameters.builderPlatform | length) > 0
  and (.SLSA.buildDefinition.resolvedDependencies | type) == "array"
  and (.SLSA.buildDefinition.resolvedDependencies | length) > 0
  and all(
    .SLSA.buildDefinition.resolvedDependencies[];
    (.uri | type) == "string"
      and (.uri | length) > 0
      and (.digest | type) == "object"
      and (.digest | length) > 0
  )
  and .SLSA.runDetails.metadata.buildkit_completeness.request == true
  and (.SLSA.runDetails.metadata.buildkit_metadata | type) == "object";

buildkit_slsa_v02 or buildkit_slsa_v1
