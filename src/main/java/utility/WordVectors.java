package utility;

import cc.mallet.types.MatrixOps;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;

public class WordVectors {
    public static double[][] readWordVectorsFile(String pathToWordVectorsFile, HashMap<String, Integer> word2IdVocabulary, HashMap<Integer, String> id2WordVocabulary)
        throws Exception
    {
        System.out.println("Reading word vectors from word-vectors file " + pathToWordVectorsFile
                + "...");

        BufferedReader br = null;
        double[][] wordVectors;
        int vocabularySize = word2IdVocabulary.size();
        try {
            br = new BufferedReader(new FileReader(pathToWordVectorsFile));
            String[] elements = br.readLine().trim().split("\\s+");
            int vectorSize = elements.length - 1;
            wordVectors = new double[vocabularySize][vectorSize];
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
                    throw new Exception();
                }
            }
            return wordVectors;
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        return new double[0][0];
    }

}
