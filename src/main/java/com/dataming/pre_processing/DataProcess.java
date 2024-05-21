package com.dataming.pre_processing;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;

import static java.lang.Math.max;

public class DataProcess {
    public DataProcess() {
    }

    public static Instances read_arff_dataset(String data_path) throws Exception {

        return new DataSource(data_path).getDataSet();
    }

    public static Instances read_csv_dataset(String data_path) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(data_path));
        return loader.getDataSet();
    }

    public static void save_csv2arff(String csv_path, String arff_path) throws Exception {
        /*
         * Convert csv file to arff file
         * Params:
         * csv_path: path to read csv file
         * arff_path: path to save arff file
         */
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csv_path));
        Instances data = loader.getDataSet();

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(arff_path));
        saver.writeBatch();
    }

    public static Instances removeColumn(Instances data, int index) {
        Instances newData = new Instances(data);
        newData.deleteAttributeAt(index);
        return newData;
    }

    public static Instances removeColumn(Instances data, String name) {
        Instances newData = new Instances(data);
        newData.deleteAttributeAt(data.attribute(name).index());
        return newData;
    }

    public static Instances fixMissingValues(Instances filteredData) {
        for (int att_idx = 0; att_idx < filteredData.numAttributes(); att_idx++) {
            if (att_idx == filteredData.classIndex())
                continue;
            if (!filteredData.attribute(att_idx).isNumeric())
                continue;
            ArrayList<Double> arrayList = new ArrayList<>();
            HashMap<String, Integer> occur = new HashMap<>();
            for (Instance instance : filteredData) {
                if (instance.isMissing(att_idx))
                    continue;
                arrayList.add(instance.value(att_idx));
                String temp = String.valueOf(instance.value(att_idx));
                if (!occur.containsKey(temp))
                    occur.put(temp, 0);
                occur.put(temp, occur.get(temp) + 1);
            }

            double sum = 0;
            for (Double val : arrayList)
                sum += val;
            double mean = sum / arrayList.size();

            double mode = 0, max_occur = -1;
            for (String _value : occur.keySet()) {
                if (occur.get(_value) > max_occur) {
                    max_occur = occur.get(_value);
                    mode = Double.parseDouble(_value);
                }
            }
            // System.out.println(mode);
            for (Instance instance : filteredData) {
                if (instance.isMissing(att_idx)) {
                    if (att_idx == 5)
                        instance.setValue(att_idx, mode);
                    else
                        instance.setValue(att_idx, mean);
                }
            }

        }

        return filteredData;
    }

    public static Instances numericToNominal(Instances data, String column_index) throws Exception {
        NumericToNominal numericToNominal = new NumericToNominal();
        numericToNominal.setAttributeIndices(column_index);
        numericToNominal.setInputFormat(data);
        data = NominalToBinary.useFilter(data, numericToNominal);
        return data;
    }

    public static Instances normalize(Instances data) {
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.classIndex() == i) {
                continue;
            }
            if (data.attribute(i).isNumeric()) {
                double mx = 0.0, mn = 1e9;

                for (int j = 0; j < data.numInstances(); j++) {
                    mx = max(mx, data.instance(j).value(i));
                    mn = max(mn, data.instance(j).value(i));
                }

                for (int j = 0; j < data.numInstances(); j++) {
                    double val = data.instance(j).value(i);
                    data.instance(j).setValue(i, val / mx);
                    // data.instance(j).setValue(i, (val - mn/ (mx - mn)));
                }
            }
        }

        return data;
    }

    public static void analyze_data(Instances data) {
        HashMap<String, Integer> _count = new HashMap<>();
        for (Instance instance : data) {
            String label = instance.stringValue(data.classIndex());
            if (_count.containsKey(label)) {
                _count.put(label, _count.get(label) + 1);
            } else {
                _count.put(label, 1);
            }
        }
        System.out.println(_count);
    }
}
