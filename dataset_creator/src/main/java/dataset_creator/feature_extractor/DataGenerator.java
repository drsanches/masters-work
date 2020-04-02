package dataset_creator.feature_extractor;

import java.util.ArrayList;
import java.util.List;

public class DataGenerator {

    private List<List<Double>> X = new ArrayList<>();
    private List<List<Double>> Y = new ArrayList<>();
    private int N;

    public DataGenerator(List<List<List<Double>>> dataset) {
        List<List<Double>> X = dataset.get(0);
        List<List<Double>> Y = dataset.get(1);
        for (List<Double> x: X) {
            this.X.add(new ArrayList<>(x));
        }
        for (List<Double> y: Y) {
            this.Y.add(new ArrayList<>(y));
        }
        this.N = X.size();
    }

    public void addShift(double k) {
        for (int i = 0; i < N; i++) {
            X.add(getShift(X.get(i), k));
            Y.add(Y.get(i));
        }
    }

    private List<Double> getShift(List<Double> x, double k) {
        List<Double> shiftedX = new ArrayList<>();
        for (Double feature: x) {
            shiftedX.add(feature + feature * k);
        }
        return shiftedX;
    }

    public int getSize() {
        return X.size();
    }

    public List<List<List<Double>>> getDataset() {
        List<List<List<Double>>> dataset = new ArrayList<>();
        dataset.add(X);
        dataset.add(Y);
        return dataset;
    }
}