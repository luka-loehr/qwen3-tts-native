if .SLSA.buildType == "https://mobyproject.org/buildkit@v1" then
  {
    schema: "buildkit-slsa-v0.2",
    parameters: .SLSA.invocation.parameters,
    environment: .SLSA.invocation.environment,
    metadata: .SLSA.metadata
  }
elif .SLSA.buildDefinition.buildType
  == "https://github.com/moby/buildkit/blob/master/docs/attestations/slsa-definitions.md"
then
  {
    schema: "buildkit-slsa-v1",
    externalParameters: .SLSA.buildDefinition.externalParameters,
    internalParameters: .SLSA.buildDefinition.internalParameters,
    runMetadata: .SLSA.runDetails.metadata
  }
else
  error("unsupported BuildKit provenance schema")
end
