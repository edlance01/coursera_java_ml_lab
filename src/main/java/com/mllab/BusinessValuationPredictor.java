package com.mllab;

import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.util.ArrayList;

public class BusinessValuationPredictor {

    private LinearRegression model;
    private Instances trainingData;

    public static void main(String[] args) {
        // Suppress netlib warnings for cleaner output (optional)
        java.util.logging.Logger.getLogger("com.github.fommil").setLevel(java.util.logging.Level.SEVERE);

        try {
            BusinessValuationPredictor predictor = new BusinessValuationPredictor();

            // Step 1: Load and prepare data
            System.out.println("=== Loading Business Data ===");
            predictor.loadData();

            // Step 2: Train the model
            System.out.println("\n=== Training Model ===");
            predictor.trainModel();

            // Step 3: Evaluate model
            System.out.println("\n=== Model Evaluation ===");
            predictor.evaluateModel();

            // Step 4: Make predictions
            System.out.println("\n=== Making Predictions ===");
            predictor.makePredictions();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void loadData() throws Exception {
        // Load CSV file
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("src/main/resources/business_data.csv"));
        Instances data = loader.getDataSet();

        // Remove non-numeric attributes (company_name)
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndices("1"); // Remove first attribute (company_name)
        removeFilter.setInputFormat(data);
        this.trainingData = Filter.useFilter(data, removeFilter);

        // Set the target attribute (valuation) as the class
        trainingData.setClassIndex(trainingData.numAttributes() - 1);

        System.out.println("Loaded " + trainingData.numInstances() + " business records");
        System.out.println("Features: " + (trainingData.numAttributes() - 1));
        System.out.println("Target: " + trainingData.classAttribute().name());

        // Display first few records
        System.out.println("\nSample data:");
        for (int i = 0; i < Math.min(3, trainingData.numInstances()); i++) {
            Instance instance = trainingData.instance(i);
            System.out.printf("EBITDA: $%.0f, Revenue: $%.0f, Employees: %.0f, " +
                            "Industry Multiple: %.1f, Years: %.0f, Position: %.0f → Valuation: $%.0f%n",
                    instance.value(0), instance.value(1), instance.value(2),
                    instance.value(3), instance.value(4), instance.value(5), instance.classValue());
        }
    }

    public void trainModel() throws Exception {
        // Create and configure linear regression model
        model = new LinearRegression();

        // Configure attribute selection - use SelectedTag for proper API compatibility
        model.setAttributeSelectionMethod(
                new weka.core.SelectedTag(LinearRegression.SELECTION_NONE,
                        LinearRegression.TAGS_SELECTION)
        );
        model.setEliminateColinearAttributes(false);

        // Train the model
        model.buildClassifier(trainingData);

        System.out.println("Model trained successfully!");
        System.out.println("\nModel Details:");
        System.out.println(model.toString());
    }

    public void evaluateModel() throws Exception {
        // Calculate predictions for training data to show performance
        double totalError = 0;
        double totalActual = 0;

        System.out.println("Actual vs Predicted Valuations (first 5 records):");
        System.out.println("Actual\t\tPredicted\tError\t\tError %");
        System.out.println("------\t\t---------\t-----\t\t-------");

        for (int i = 0; i < Math.min(5, trainingData.numInstances()); i++) {
            Instance instance = trainingData.instance(i);
            double actual = instance.classValue();
            double predicted = model.classifyInstance(instance);
            double error = Math.abs(actual - predicted);
            double errorPercent = (error / actual) * 100;

            totalError += error;
            totalActual += actual;

            System.out.printf("$%.0f\t\t$%.0f\t\t$%.0f\t\t%.1f%%%n",
                    actual, predicted, error, errorPercent);
        }

        double avgErrorPercent = (totalError / totalActual) * 100;
        System.out.printf("%nAverage Error: %.1f%%%n", avgErrorPercent);
    }

    public void makePredictions() throws Exception {
        // Create new business scenarios for prediction
        String[] scenarios = {
                "New Tech Startup: EBITDA $200K, Revenue $600K, 5 employees, 7 years, average position",
                "Established Consulting Firm: EBITDA $800K, Revenue $2M, 30 employees, 20 years, market leader",
                "Small Marketing Agency: EBITDA $150K, Revenue $450K, 4 employees, 3 years, weak position"
        };

        double[][] scenarioData = {
                {200000, 600000, 5, 3.5, 7, 3},    // Tech startup
                {800000, 2000000, 30, 5.0, 20, 1}, // Established consulting
                {150000, 450000, 4, 3.2, 3, 4}     // Small agency
        };

        System.out.println("Business Valuation Predictions:");
        System.out.println("================================");

        for (int i = 0; i < scenarios.length; i++) {
            // Create new instance
            Instance newBusiness = new DenseInstance(trainingData.numAttributes());
            newBusiness.setDataset(trainingData);

            // Set attribute values
            for (int j = 0; j < scenarioData[i].length; j++) {
                newBusiness.setValue(j, scenarioData[i][j]);
            }

            // Make prediction
            double predictedValue = model.classifyInstance(newBusiness);
            double ebitdaMultiple = predictedValue / scenarioData[i][0];

            System.out.println("\nScenario " + (i + 1) + ": " + scenarios[i]);
            System.out.printf("Predicted Valuation: $%.0f%n", predictedValue);
            System.out.printf("EBITDA Multiple: %.1fx%n", ebitdaMultiple);

            // Business insights
            if (ebitdaMultiple > 4.5) {
                System.out.println("Assessment: Premium valuation - strong business metrics");
            } else if (ebitdaMultiple > 3.5) {
                System.out.println("Assessment: Fair market valuation");
            } else {
                System.out.println("Assessment: Below market - consider business improvements");
            }
        }
    }
}