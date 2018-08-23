package models;

import utility.WordVectors;

import java.util.ArrayList;
import java.util.List;

public class TopicModel {
    public double alpha; // Hyper-parameter alpha
    public double beta; // Hyper-parameter alpha
    public double betaSum; // beta * wordVectors.getVocabularySize()

    public double[][] topicVectors;// Vector representations for topics
    public int numTopics; // Number of topics
    public int topWords; // Number of most probable words for each topic

    public double lambda; // Mixture weight value
    public int numInitIterations;
    public int numIterations; // Number of EM-style sampling iterations
    protected WordVectors wordVectors;
    protected ArrayList<List<Integer>> topicAssignments;
    public double[][] dotProductValues;
    public double[][] expDotProductValues;
    public double[] sumExpValues; // Partition function values

    // wordVectors.getNumDocuments() * numTopics matrix
    // Given a document: number of its words assigned to each topic
    // Number of words in every document
    // numTopics * wordVectors.getVocabularySize() matrix
    // Given a topic: number of times a word type generated from the topic by
    // the Dirichlet multinomial component
    public int[][] topicWordCount;
    // Total number of words generated from each topic by the Dirichlet
    // multinomial component
    public int[] sumTopicWordCount;
    // numTopics * wordVectors.getVocabularySize() matrix
    // Given a topic: number of times a word type generated from the topic by
    // the latent feature component
    public int[][] topicWordCountLF;
    // Total number of words generated from each topic by the latent feature
    // component
    public int[] sumTopicWordCountLF;

    // Double array used to sample a topic
    public double[] multiPros;
    // Path to the directory containing the corpus
    public String folderPath;
    // Path to the topic modeling corpus
    public String corpusPath;
    public String vectorFilePath;

    public final double l2Regularizer = 0.01; // L2 regularizer value for learning topic vectors
    public final double tolerance = 0.05; // Tolerance value for LBFGS convergence

}
