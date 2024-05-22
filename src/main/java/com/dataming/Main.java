package com.dataming;

import com.dataming.pre_processing.DataProcess;
import com.github.freva.asciitable.AsciiTable;
import com.github.freva.asciitable.Column;
import com.github.freva.asciitable.ColumnData;
import com.github.freva.asciitable.HorizontalAlign;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Standardize;
import weka.classifiers.trees.RandomTree;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Main {
    public static void main(String[] args) throws Exception {
        String csvFilePath = "./raw_data/wind_dataset_cleaned.csv";
        String preprocessedArffFilePath = "./src/main/resources/clean_data/wind_dataset_preprocessed.arff";
        DataProcess.save_csv2arff(csvFilePath, preprocessedArffFilePath);

        DataSource source = new DataSource("./src/main/resources/clean_data/wind_dataset_preprocessed.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(1);

        // Remove the first attribute
        Remove remove = new Remove();
        remove.setAttributeIndices("1");
        remove.setInputFormat(dataset);
        Instances newData = weka.filters.Filter.useFilter(dataset, remove);

        // Standardize the data
        Standardize standardize = new Standardize();
        standardize.setInputFormat(newData);
        Instances standardizedData = weka.filters.Filter.useFilter(newData, standardize);


        RandomForest rf = new RandomForest();
        RandomTree rt = new RandomTree();
        SMOreg smo = new SMOreg();
        LinearRegression linearRegression = new LinearRegression();
        M5P m5p = new M5P();
        MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron();


        ArrayList<Classifier> classifiers = new ArrayList<>();

        classifiers.add(rf);
        classifiers.add(rt);
        classifiers.add(smo);
        classifiers.add(linearRegression);
        classifiers.add(m5p);
        classifiers.add(multilayerPerceptron);

        String[] header = {"", "RandomForest", "RandomTree", "SMOreg", "LinearRegression", "M5P", "MultilayerPerceptron"};
        String[][] results = {
                {"Correlation coefficient:", "", "", "", "", "", ""},
                {"Mean absolute error:", "", "", "", "", "", ""},
                {"Root mean squared error:", "", "", "", "", "", ""},
                {"Relative absolute error:", "", "", "", "", "", ""},
                {"Root relative squared error:", "", "", "", "", "", ""},
                {"Total Number of Instances:", "", "", "", "", "", ""},
                {"Time taken:", "", "", "", "", "", ""}
        };

        int index = 1;
        double Score = Double.NEGATIVE_INFINITY;
        Classifier bestClassifier = null;

        for (Classifier classifier : classifiers) {
            long startTime = System.currentTimeMillis();
            classifier.buildClassifier(standardizedData);
            Evaluation eval = new Evaluation(standardizedData);
            eval.crossValidateModel(classifier, standardizedData, 10, new Random(1));
            long endTime = System.currentTimeMillis();
            long timeTaken = endTime - startTime;
//            System.out.println("\nClassifier: " + classifier.getClass().getSimpleName());
//            System.out.println("=====================================");
            results[0][index] = String.format("%.5f", eval.correlationCoefficient());
            results[1][index] = String.format("%.5f", eval.meanAbsoluteError());
            results[2][index] = String.format("%.5f", eval.rootMeanSquaredError());
            results[3][index] = String.format("%.5f", eval.relativeAbsoluteError()) + "%";
            results[4][index] = String.format("%.5f", eval.rootRelativeSquaredError()) + "%";
            results[5][index] = String.valueOf((int) eval.numInstances());
            results[6][index] = timeTaken + " ms";
            index++;

            if (eval.correlationCoefficient() > Score) {
                Score = eval.correlationCoefficient();
                bestClassifier = classifier;
            }
        }
//
        System.out.println(AsciiTable.getTable(header, results));

        System.out.println("=======================================================================");
        System.out.println("Best Classifier: " + bestClassifier.getClass().getSimpleName());
        System.out.println("Score: " + Score);
    }
}