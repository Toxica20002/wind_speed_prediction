package com.dataming.pre_processing;

import weka.core.Instances;
import weka.core.Instance;

public class DataCleaner {

    public static Instances removeNonNumericRows(Instances data) {
        Instances cleanedData = new Instances(data, 0); // Create an empty dataset with the same structure

        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            boolean isValid = true;

            for (int j = 0; j < data.numAttributes(); j++) {
                // Skip if attribute is nominal or string
                if (data.attribute(j).isNominal() || data.attribute(j).isString()) continue;

                // Check if the value is missing or not a number for numeric attributes
                if (instance.isMissing(j) ) {
                    isValid = false;
                    break;
                }

                if (!isNumeric(instance.toString(j))) {
                    isValid = false;
                    break;
                }
            }

            if (isValid) {
                cleanedData.add(instance);
            }
        }

        return cleanedData;
    }

    private static boolean isNumeric(String str) {
        try {
            Double.parseDouble(str);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }
}