/*
 * (c) Copyright 2018 Palantir Technologies Inc. All rights reserved.
 */

package myproject;

import com.palantir.transforms.lang.java.api.Pipeline;
import com.palantir.transforms.lang.java.api.PipelineDefiner;

public final class MyPipelineDefiner implements PipelineDefiner {

    @Override
    public void define(Pipeline pipeline) {
        // This automatically registers Transforms in the provided package.
        // Automatic registration is the recommended approach for most usages.
        pipeline.autoBindFromPackage("myproject.datasets");

        //        // For manually registered Transforms, information about your input/output
        //        // datasets is provided in your "PipelineDefiner" implementation.
        //
        //        // This is a sample manual registration for a low-level Transform.
        //        LowLevelTransform lowLevelManualTransform = LowLevelTransform.builder()
        //                // Pass in the compute function to use. Here, "LowLevelManualFunction" corresponds
        //                // to the class name for a compute function for a low-level Transform.
        //                .computeFunctionInstance(new LowLevelManualFunction())
        //                // Pass in the input dataset(s) to use.
        //                // "myInput" corresponds to an input parameter for your compute function.
        //                .putParameterToInputAlias("myInput", "/path/to/input/dataset")
        //                // Pass in the output dataset(s) to use.
        //                // "myOutput" corresponds to an input parameter for your compute function.
        //                .putParameterToOutputAlias("myOutput", "/path/to/output/dataset")
        //                .build();
        //        pipeline.register(lowLevelManualTransform);
        //
        //        // This is a sample manual registration for a high-level Transform.
        //        HighLevelTransform highLevelManualTransform = HighLevelTransform.builder()
        //                // Pass in the compute function to use. Here, "HighLevelManualFunction" corresponds
        //                // to the class name for a compute function for a high-level Transform.
        //                .computeFunctionInstance(new HighLevelManualFunction())
        //                // Pass in the input dataset(s) to use.
        //                // "myInput" corresponds to an input parameter for your compute function.
        //                .putParameterToInputAlias("myInput", "/path/to/input/dataset")
        //                // Pass in the output dataset to use.
        //                .returnedAlias("/path/to/output/dataset")
        //                .build();
        //        pipeline.register(highLevelManualTransform);
    }
}
