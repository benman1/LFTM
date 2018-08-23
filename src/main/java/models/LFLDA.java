package models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import utility.FuncUtils;
import utility.LBFGS;
import utility.Parallel;
import cc.mallet.optimize.InvalidOptimizableException;
import cc.mallet.optimize.Optimizer;
import utility.WordVectors;

/**
 * Implementation of the LF-LDA latent feature topic model, using collapsed Gibbs sampling, as
 * described in:
 * 
 * Dat Quoc Nguyen, Richard Billingsley, Lan Du and Mark Johnson. 2015. Improving Topic Models with
 * Latent Feature Word Representations. Transactions of the Association for Computational
 * Linguistics, vol. 3, pp. 299-313.
 * 
 * @author Dat Quoc Nguyen
 */

public class LFLDA extends TopicModel
{
    private final int[] sumDocTopicCount;
    public double alpha; // Hyper-parameter alpha
    public double beta; // Hyper-parameter alpha
    public double alphaSum; // alpha * numTopics

    public WordVectors wordVectors; // Vector representations for words

    public String expName = "LFLDA";
    public String orgExpName = "LFLDA";
    public String tAssignsFilePath = "";
    public int savestep = 0;
    private int[][] docTopicCount;

    public LFLDA(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
            double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
            int inNumIterations, int inTopWords)
        throws Exception
    {
        this(pathToCorpus, pathToWordVectorsFile, inNumTopics, inAlpha, inBeta, inLambda,
                inNumInitIterations, inNumIterations, inTopWords, "LFLDA");
    }

    public LFLDA(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
            double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
            int inNumIterations, int inTopWords, String inExpName)
        throws Exception
    {
        this(pathToCorpus, pathToWordVectorsFile, inNumTopics, inAlpha, inBeta, inLambda,
                inNumInitIterations, inNumIterations, inTopWords, inExpName, "", 0);
    }

    public LFLDA(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
            double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
            int inNumIterations, int inTopWords, String inExpName, String pathToTAfile)
        throws Exception
    {
        this(pathToCorpus, pathToWordVectorsFile, inNumTopics, inAlpha, inBeta, inLambda,
                inNumInitIterations, inNumIterations, inTopWords, inExpName, pathToTAfile, 0);
    }

    public LFLDA(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
            double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
            int inNumIterations, int inTopWords, String inExpName, int inSaveStep)
        throws Exception
    {
        this(pathToCorpus, pathToWordVectorsFile, inNumTopics, inAlpha, inBeta, inLambda,
                inNumInitIterations, inNumIterations, inTopWords, inExpName, "", inSaveStep);
    }

    public LFLDA(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
            double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
            int inNumIterations, int inTopWords, String inExpName, String pathToTAfile,
            int inSaveStep)
        throws Exception
    {
        alpha = inAlpha;
        beta = inBeta;
        lambda = inLambda;
        numTopics = inNumTopics;
        numIterations = inNumIterations;
        numInitIterations = inNumInitIterations;
        topWords = inTopWords;
        savestep = inSaveStep;
        expName = inExpName;
        orgExpName = expName;
        vectorFilePath = pathToWordVectorsFile;
        corpusPath = pathToCorpus;
        folderPath = pathToCorpus.substring(0,
                Math.max(pathToCorpus.lastIndexOf("/"), pathToCorpus.lastIndexOf("\\")) + 1);

        System.out.println("Reading topic modeling corpus: " + pathToCorpus);

        wordVectors = new WordVectors(corpusPath, vectorFilePath);

        docTopicCount = new int[wordVectors.getNumDocuments()][numTopics];
        sumDocTopicCount = new int[wordVectors.getNumDocuments()];
        topicWordCount = new int[numTopics][wordVectors.getVocabularySize()];
        sumTopicWordCount = new int[numTopics];
        topicWordCountLF = new int[numTopics][wordVectors.getVocabularySize()];
        sumTopicWordCountLF = new int[numTopics];

        multiPros = new double[numTopics * 2];
        for (int i = 0; i < numTopics * 2; i++) {
            multiPros[i] = 1.0 / numTopics;
        }

        alphaSum = numTopics * alpha;
        betaSum = wordVectors.getVocabularySize() * beta;

        topicVectors = new double[numTopics][wordVectors.getVectorSize()];
        dotProductValues = new double[numTopics][wordVectors.getVocabularySize()];
        expDotProductValues = new double[numTopics][wordVectors.getVocabularySize()];
        sumExpValues = new double[numTopics];

        System.out.println("Number of topics: " + numTopics);
        System.out.println("alpha: " + alpha);
        System.out.println("beta: " + beta);
        System.out.println("lambda: " + lambda);
        System.out.println("Number of initial sampling iterations: " + numInitIterations);
        System.out.println("Number of EM-style sampling iterations for the LF-LDA model: "
                + numIterations);
        System.out.println("Number of top topical words: " + topWords);

        tAssignsFilePath = pathToTAfile;
        if (tAssignsFilePath.length() > 0)
            initialize(tAssignsFilePath);
        else
            initialize();

    }

    public void initialize()
        throws IOException
    {
        System.out.println("Randomly initialzing topic assignments ...");
        topicAssignments = new ArrayList<List<Integer>>();

        for (int docId = 0; docId < wordVectors.getNumDocuments(); docId++) {
            List<Integer> topics = new ArrayList<Integer>();
            int docSize = wordVectors.getCorpus().get(docId).size();
            for (int j = 0; j < docSize; j++) {
                int wordId = wordVectors.getCorpus().get(docId).get(j);

                int subtopic = FuncUtils.nextDiscrete(multiPros);
                int topic = subtopic % numTopics;
                if (topic == subtopic) { // Generated from the latent feature component
                    topicWordCountLF[topic][wordId] += 1;
                    sumTopicWordCountLF[topic] += 1;
                }
                else {// Generated from the Dirichlet multinomial component
                    topicWordCount[topic][wordId] += 1;
                    sumTopicWordCount[topic] += 1;
                }
                docTopicCount[docId][topic] += 1;
                sumDocTopicCount[docId] += 1;

                topics.add(subtopic);
            }
            topicAssignments.add(topics);
        }
    }

    public void initialize(String pathToTopicAssignmentFile)
    {
        System.out.println("Reading topic-assignment file: " + pathToTopicAssignmentFile);

        topicAssignments = new ArrayList<List<Integer>>();

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(pathToTopicAssignmentFile));
            int docId = 0;
            int numWords = 0;
            for (String line; (line = br.readLine()) != null;) {
                String[] strTopics = line.trim().split("\\s+");
                List<Integer> topics = new ArrayList<Integer>();
                for (int j = 0; j < strTopics.length; j++) {
                    int wordId = wordVectors.getCorpus().get(docId).get(j);

                    int subtopic = new Integer(strTopics[j]);
                    int topic = subtopic % numTopics;

                    if (topic == subtopic) { // Generated from the latent feature component
                        topicWordCountLF[topic][wordId] += 1;
                        sumTopicWordCountLF[topic] += 1;
                    }
                    else {// Generated from the Dirichlet multinomial component
                        topicWordCount[topic][wordId] += 1;
                        sumTopicWordCount[topic] += 1;
                    }
                    docTopicCount[docId][topic] += 1;
                    sumDocTopicCount[docId] += 1;

                    topics.add(subtopic);
                    numWords++;
                }
                topicAssignments.add(topics);
                docId++;
            }

            if ((docId != wordVectors.getNumDocuments()) || (numWords != wordVectors.getNumWordsInCorpus())) {
                System.out
                        .println("The topic modeling corpus and topic assignment file are not consistent!!!");
                throw new Exception();
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void inference()
        throws IOException
    {
        System.out.println("Running Gibbs sampling inference: ");

        for (int iter = 1; iter <= numInitIterations; iter++) {

            System.out.println("\tInitial sampling iteration: " + (iter));

            sampleSingleInitialIteration();
        }

        for (int iter = 1; iter <= numIterations; iter++) {

            System.out.println("\tLFLDA sampling iteration: " + (iter));

            optimizeTopicVectors();

            sampleSingleIteration();

            if ((savestep > 0) && (iter % savestep == 0) && (iter < numIterations)) {
                System.out.println("\t\tSaving the output from the " + iter + "^{th} sample");
                expName = orgExpName + "-" + iter;
                write();
            }
        }
        expName = orgExpName;

        writeParameters();
        System.out.println("Writing output from the last sample ...");
        write();

        System.out.println("Sampling completed!");
    }

    public void optimizeTopicVectors()
    {
        System.out.println("\t\tEstimating topic vectors ...");
        sumExpValues = new double[numTopics];
        dotProductValues = new double[numTopics][wordVectors.getVocabularySize()];
        expDotProductValues = new double[numTopics][wordVectors.getVocabularySize()];

        Parallel.loop(numTopics, new Parallel.LoopInt()
        {
            @Override
            public void compute(int topic)
            {
                int rate = 1;
                boolean check = true;
                while (check) {
                    double l2Value = l2Regularizer * rate;
                    try {
                        TopicVectorOptimizer optimizer = new TopicVectorOptimizer(
                                topicVectors[topic], topicWordCountLF[topic], wordVectors.getWordVectors(), l2Value);

                        Optimizer gd = new LBFGS(optimizer, tolerance);
                        gd.optimize(600);
                        optimizer.getParameters(topicVectors[topic]);
                        sumExpValues[topic] = optimizer.computePartitionFunction(
                                dotProductValues[topic], expDotProductValues[topic]);
                        check = false;

                        if (sumExpValues[topic] == 0 || Double.isInfinite(sumExpValues[topic])) {
                            double max = -1000000000.0;
                            for (int index = 0; index < wordVectors.getVocabularySize(); index++) {
                                if (dotProductValues[topic][index] > max)
                                    max = dotProductValues[topic][index];
                            }
                            for (int index = 0; index < wordVectors.getVocabularySize(); index++) {
                                expDotProductValues[topic][index] = Math
                                        .exp(dotProductValues[topic][index] - max);
                                sumExpValues[topic] += expDotProductValues[topic][index];
                            }
                        }
                    }
                    catch (InvalidOptimizableException e) {
                        e.printStackTrace();
                        check = true;
                    }
                    rate = rate * 10;
                }
            }
        });
    }

    public void sampleSingleIteration()
    {
        for (int dIndex = 0; dIndex < wordVectors.getNumDocuments(); dIndex++) {
            int docSize = wordVectors.getCorpus().get(dIndex).size();
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                // Get current word
                int word = wordVectors.getCorpus().get(dIndex).get(wIndex);// wordID
                int subtopic = topicAssignments.get(dIndex).get(wIndex);
                int topic = subtopic % numTopics;

                docTopicCount[dIndex][topic] -= 1;
                if (subtopic == topic) {
                    topicWordCountLF[topic][word] -= 1;
                    sumTopicWordCountLF[topic] -= 1;
                }
                else {
                    topicWordCount[topic][word] -= 1;
                    sumTopicWordCount[topic] -= 1;
                }

                // Sample a pair of topic z and binary indicator variable s
                for (int tIndex = 0; tIndex < numTopics; tIndex++) {

                    multiPros[tIndex] = (docTopicCount[dIndex][tIndex] + alpha) * lambda
                            * expDotProductValues[tIndex][word] / sumExpValues[tIndex];

                    multiPros[tIndex + numTopics] = (docTopicCount[dIndex][tIndex] + alpha)
                            * (1 - lambda) * (topicWordCount[tIndex][word] + beta)
                            / (sumTopicWordCount[tIndex] + betaSum);

                }
                subtopic = FuncUtils.nextDiscrete(multiPros);
                topic = subtopic % numTopics;

                docTopicCount[dIndex][topic] += 1;
                if (subtopic == topic) {
                    topicWordCountLF[topic][word] += 1;
                    sumTopicWordCountLF[topic] += 1;
                }
                else {
                    topicWordCount[topic][word] += 1;
                    sumTopicWordCount[topic] += 1;
                }
                // Update topic assignments
                topicAssignments.get(dIndex).set(wIndex, subtopic);
            }
        }
    }

    public void sampleSingleInitialIteration()
    {
        for (int dIndex = 0; dIndex < wordVectors.getNumDocuments(); dIndex++) {
            int docSize = wordVectors.getCorpus().get(dIndex).size();
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                int word = wordVectors.getCorpus().get(dIndex).get(wIndex);// wordID
                int subtopic = topicAssignments.get(dIndex).get(wIndex);
                int topic = subtopic % numTopics;

                docTopicCount[dIndex][topic] -= 1;
                if (subtopic == topic) { // LF(w|t) + LDA(t|d)
                    topicWordCountLF[topic][word] -= 1;
                    sumTopicWordCountLF[topic] -= 1;
                }
                else { // LDA(w|t) + LDA(t|d)
                    topicWordCount[topic][word] -= 1;
                    sumTopicWordCount[topic] -= 1;
                }

                // Sample a pair of topic z and binary indicator variable s
                for (int tIndex = 0; tIndex < numTopics; tIndex++) {

                    multiPros[tIndex] = (docTopicCount[dIndex][tIndex] + alpha) * lambda
                            * (topicWordCountLF[tIndex][word] + beta)
                            / (sumTopicWordCountLF[tIndex] + betaSum);

                    multiPros[tIndex + numTopics] = (docTopicCount[dIndex][tIndex] + alpha)
                            * (1 - lambda) * (topicWordCount[tIndex][word] + beta)
                            / (sumTopicWordCount[tIndex] + betaSum);

                }
                subtopic = FuncUtils.nextDiscrete(multiPros);
                topic = subtopic % numTopics;

                docTopicCount[dIndex][topic] += 1;
                if (topic == subtopic) {
                    topicWordCountLF[topic][word] += 1;
                    sumTopicWordCountLF[topic] += 1;
                }
                else {
                    topicWordCount[topic][word] += 1;
                    sumTopicWordCount[topic] += 1;
                }
                // Update topic assignments
                topicAssignments.get(dIndex).set(wIndex, subtopic);
            }

        }
    }

    public void writeParameters()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".paras"));
        writer.write("-model" + "\t" + "LFLDA");
        writer.write("\n-corpus" + "\t" + corpusPath);
        writer.write("\n-vectors" + "\t" + vectorFilePath);
        writer.write("\n-ntopics" + "\t" + numTopics);
        writer.write("\n-alpha" + "\t" + alpha);
        writer.write("\n-beta" + "\t" + beta);
        writer.write("\n-lambda" + "\t" + lambda);
        writer.write("\n-initers" + "\t" + numInitIterations);
        writer.write("\n-niters" + "\t" + numIterations);
        writer.write("\n-twords" + "\t" + topWords);
        writer.write("\n-name" + "\t" + expName);
        if (tAssignsFilePath.length() > 0)
            writer.write("\n-initFile" + "\t" + tAssignsFilePath);
        if (savestep > 0)
            writer.write("\n-sstep" + "\t" + savestep);

        writer.close();
    }

    public void writeDictionary()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
                + ".vocabulary"));
        for (String word : wordVectors.getWord2IdVocabulary().keySet()) {
            writer.write(word + " " + wordVectors.getWord2IdVocabulary().get(word) + "\n");
        }
        writer.close();
    }

    public void writeIDbasedCorpus()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
                + ".IDcorpus"));
        for (int dIndex = 0; dIndex < wordVectors.getNumDocuments(); dIndex++) {
            int docSize = wordVectors.getCorpus().get(dIndex).size();
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                writer.write(wordVectors.getCorpus().get(dIndex).get(wIndex) + " ");
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void writeTopicAssignments()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
                + ".topicAssignments"));
        for (int dIndex = 0; dIndex < wordVectors.getNumDocuments(); dIndex++) {
            int docSize = wordVectors.getCorpus().get(dIndex).size();
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                writer.write(topicAssignments.get(dIndex).get(wIndex) + " ");
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void writeTopicVectors()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
                + ".topicVectors"));
        for (int i = 0; i < numTopics; i++) {
            for (int j = 0; j < wordVectors.getVectorSize(); j++)
                writer.write(topicVectors[i][j] + " ");
            writer.write("\n");
        }
        writer.close();
    }

    public void writeTopTopicalWords()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
                + ".topWords"));

        for (int tIndex = 0; tIndex < numTopics; tIndex++) {
            writer.write("Topic" + tIndex + ":");

            Map<Integer, Double> topicWordProbs = new TreeMap<Integer, Double>();
            for (int wIndex = 0; wIndex < wordVectors.getVocabularySize(); wIndex++) {

                double pro = lambda * expDotProductValues[tIndex][wIndex] / sumExpValues[tIndex]
                        + (1 - lambda) * (topicWordCount[tIndex][wIndex] + beta)
                        / (sumTopicWordCount[tIndex] + betaSum);

                topicWordProbs.put(wIndex, pro);
            }
            topicWordProbs = FuncUtils.sortByValueDescending(topicWordProbs);

            Set<Integer> mostLikelyWords = topicWordProbs.keySet();
            int count = 0;
            for (Integer index : mostLikelyWords) {
                if (count < topWords) {
                    writer.write(" " + wordVectors.getId2WordVocabulary().get(index));
                    count += 1;
                }
                else {
                    writer.write("\n\n");
                    break;
                }
            }
        }
        writer.close();
    }

    public void writeTopicWordPros()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".phi"));
        for (int t = 0; t < numTopics; t++) {
            for (int w = 0; w < wordVectors.getVocabularySize(); w++) {
                double pro = lambda * expDotProductValues[t][w] / sumExpValues[t] + (1 - lambda)
                        * (topicWordCount[t][w] + beta) / (sumTopicWordCount[t] + betaSum);
                writer.write(pro + " ");
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void writeDocTopicPros()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".theta"));

        for (int i = 0; i < wordVectors.getNumDocuments(); i++) {
            for (int j = 0; j < numTopics; j++) {
                double pro = (docTopicCount[i][j] + alpha) / (sumDocTopicCount[i] + alphaSum);
                writer.write(pro + " ");
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void write()
        throws IOException
    {
        writeTopTopicalWords();
        writeDocTopicPros();
        writeTopicAssignments();
        writeTopicWordPros();
    }

    public static void main(String args[])
        throws Exception
    {
        LFLDA lflda = new LFLDA("test/corpus.txt", "test/wordVectors.txt", 4, 0.1, 0.01, 0.6, 2000,
                200, 20, "testLFLDA");
        lflda.writeParameters();
        lflda.inference();
    }
}
