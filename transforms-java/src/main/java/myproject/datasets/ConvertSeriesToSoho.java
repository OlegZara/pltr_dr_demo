package myproject.datasets;

import com.palantir.transforms.lang.java.api.Compute;
import com.palantir.transforms.lang.java.api.FoundryInput;
import com.palantir.transforms.lang.java.api.FoundryOutput;
import com.palantir.transforms.lang.java.api.Input;
import com.palantir.transforms.lang.java.api.Output;
import com.palantir.foundry.spark.api.DatasetFormatSettings;

/**
* This is an example high-level Transform intended for automatic registration.
*/
public final class ConvertSeriesToSoho {

    @Compute
    public void myComputeFunction(
            @Input("ri.foundry.main.dataset.19ab7962-d363-4594-a910-4d2b551381c3") FoundryInput myInput,
            @Output("/DataRobot/Foundry DataRobot Demo/Demand Forcasting/myproject/datasets/predictions-series-soho") FoundryOutput myOutput) {
        myOutput.getDataFrameWriter(myInput.asDataFrame().read())
            .setFormatSettings(DatasetFormatSettings.builder()
                .format("soho")
                .build())
            .write();
    }
}
