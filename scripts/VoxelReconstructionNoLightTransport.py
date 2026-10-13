import falcor

def render_graph_Pass():
    g = falcor.RenderGraph("VoxelReconstructionNoLightTransport")

    VoxelReconstructionNoLightTransport_pass = falcor.createPass("VoxelReconstructionNoLightTransport")
    accumulate_pass = falcor.createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single",'maxFrameCount': 1024})
    # Evaluate the accumulated linear RGB against the white-background reference.
    # Use with reconstruction disabled and Use ReferenceCamera enabled.
    flip_pass = falcor.createPass("FLIPPass", {
        "enabled": True,
        "isHDR": False,
        "useMagma": True,
        "clampInput": True,
        "computePooledFLIPValues": True,
        "useRealMonitorInfo": False,
    })
    result_pass = falcor.createPass("ReconstructionResultPass")
    g.addPass(VoxelReconstructionNoLightTransport_pass,"VoxelReconstruction")
    g.addPass(accumulate_pass,"AccumulatePass")
    g.addPass(flip_pass, "FLIPPass")
    g.addPass(result_pass, "ResultSave")

    g.addEdge("VoxelReconstruction.color","AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "FLIPPass.testImage")
    g.addEdge("VoxelReconstruction.dummy", "FLIPPass.referenceImage")
    g.addEdge("AccumulatePass.output", "ResultSave.output")
    g.addEdge("VoxelReconstruction.dummy", "ResultSave.ref")
    g.addEdge("FLIPPass.errorMapDisplay", "ResultSave.flip")
    # An execution edge keeps this output-less, on-demand writer in the graph.
    g.addEdge("FLIPPass", "ResultSave")
    g.markOutput("VoxelReconstruction.color")
    g.markOutput("AccumulatePass.output")
    g.markOutput("VoxelReconstruction.dummy")
    g.markOutput("FLIPPass.errorMapDisplay")
    return g

Graph = render_graph_Pass()
try:
    m.addGraph(Graph)
except NameError:
    pass
