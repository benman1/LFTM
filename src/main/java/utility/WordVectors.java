package utility;

import cc.mallet.types.MatrixOps;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class WordVectors {

    public int getVectorSize() {
        return vectorSize;
    }

    private int vectorSize;

    public HashMap<String, Integer> getWord2IdVocabulary() {
        return word2IdVocabulary;
    }

    public HashMap<Integer, String> getId2WordVocabulary() {
        return id2WordVocabulary;
    }

    private HashMap<String, Integer> word2IdVocabulary;
    private HashMap<Integer, String> id2WordVocabulary;

    public ArrayList<List<Integer>> getCorpus() {
        return corpus;
    }

    private ArrayList<List<Integer>> corpus;

    public int getNumDocuments() {
        return numDocuments;
    }

    private int numDocuments;

    public int getNumWordsInCorpus() {
        return numWordsInCorpus;
    }

    private int numWordsInCorpus;

    public double[][] getWordVectors() {
        return wordVectors;
    }

    private double[][] wordVectors;

    public int getVocabularySize() {
        return vocabularySize;
    }

    private int vocabularySize;

    public WordVectors(String pathToCorpus, String pathToWordVectorsFile) {
        readCorpus(pathToCorpus);
        readWordVectorsFile(pathToWordVectorsFile);
    }


    public void readCorpus(String pathToCorpus) {
        word2IdVocabulary = new HashMap<String, Integer>();
        id2WordVocabulary = new HashMap<Integer, String>();
        corpus = new ArrayList<List<Integer>>();
        numDocuments = 0;
        numWordsInCorpus = 0;

        BufferedReader br = null;
        try {
            int indexWord = -1;
            br = new BufferedReader(new FileReader(pathToCorpus));
            for (String doc; (doc = br.readLine()) != null;) {

                if (doc.trim().length() == 0)
                    continue;

                String[] words = doc.trim().split("\\s+");
                List<Integer> document = new ArrayList<Integer>();

                for (String word : words) {
                    if (word2IdVocabulary.containsKey(word)) {
                        document.add(word2IdVocabulary.get(word));
                    }
                    else {
                        indexWord += 1;
                        word2IdVocabulary.put(word, indexWord);
                        id2WordVocabulary.put(indexWord, word);
                        document.add(indexWord);
                    }
                }

                numDocuments++;
                numWordsInCorpus += document.size();
                corpus.add(document);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        vocabularySize = word2IdVocabulary.size();
    }

    public void readWordVectorsFile(String pathToWordVectorsFile) {
        System.out.println("Reading word vectors from word-vectors file " + pathToWordVectorsFile
                + "...");

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(pathToWordVectorsFile));
            String[] elements = br.readLine().trim().split("\\s+");
            this.vectorSize = elements.length - 1;
            this.wordVectors = new double[vocabularySize][vectorSize];
            String word = elements[0];
            if (word2IdVocabulary.containsKey(word)) {
                for (int j = 0; j < vectorSize; j++) {
                    wordVectors[word2IdVocabulary.get(word)][j] = new Double(elements[j + 1]);
                }
            }
            for (String line; (line = br.readLine()) != null;) {
                elements = line.trim().split("\\s+");
                word = elements[0];
                if (word2IdVocabulary.containsKey(word)) {
                    for (int j = 0; j < vectorSize; j++) {
                        wordVectors[word2IdVocabulary.get(word)][j] = new Double(elements[j + 1]);
                    }
                }
            }

            for (int i = 0; i < vocabularySize; i++) {
                if (MatrixOps.absNorm(wordVectors[i]) == 0.0) {
                    System.out.println("The word \"" + id2WordVocabulary.get(i)
                            + "\" doesn't have a corresponding vector!!!");
                    // throw new Exception();
                }
            }
            System.out.println("Corpus size: " + getNumDocuments() + " docs, " + getNumWordsInCorpus() + " words");
            System.out.println("Vocabulary size: " + getVocabularySize());
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

}
