package myproject;

//import static org.assertj.core.api.Assertions.assertThat;
//
//import com.palantir.transforms.lang.java.testing.api.SparkSessionResource;
//import java.util.Arrays;
//import java.util.List;
//import org.apache.spark.sql.Dataset;
//import org.apache.spark.sql.Encoders;
//import org.apache.spark.sql.Row;
//import org.junit.ClassRule;
//import org.junit.Test;
//
///*
// * This is an example unit test using SparkSessionResource rule.
// */
//public class MyTests {
//
//    @ClassRule
//    public static SparkSessionResource sparkSessionResource = new SparkSessionResource();
//
//    @Test
//    public void someUnitTest() {
//        List<String> jsonData = Arrays.asList(
//                "{\"name\":\"Sarah\",\"address\":{\"city\":\"New York\",\"state\":\"NY\"}}");
//        Dataset<String> anotherPeopleDataset = sparkSessionResource.get().createDataset(jsonData, Encoders.STRING());
//        Dataset<Row> anotherPeople = sparkSessionResource.get().read().json(anotherPeopleDataset);
//        assertThat(anotherPeople.collectAsList()).hasSize(1);
//    }
//}
