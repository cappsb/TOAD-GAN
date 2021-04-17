/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package distance.test;

/**
 *
 * @author developer
 */


import ch.idsia.mario.engine.level.Level;
import distance.convolution.ConvNTuple;
import distance.kl.KLDiv;
import distance.pattern.PatternCount;
import static distance.test.KLDivTest.getConvNTuple;
import static distance.util.MarioReader.readLevel;
import java.io.FileInputStream;

import static edu.southwestern.tasks.mario.level.LevelParser.tiles;
import java.io.File;
import java.util.ArrayList;

import java.util.List;
import java.util.Scanner;
import static distance.util.MarioReader.readLevel;
import edu.southwestern.parameters.Parameters;
import edu.southwestern.tasks.mario.gan.MarioGANUtil;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.PrintWriter;
import java.util.HashMap;

public class InteractiveMarioKLDivAnalysis {
    static String trainingInputPath = "data/mario/levelsNewEncodingTest/";
    static String interactiveInputPath= "data/mario/MarioInteractiveLast/";
    static String initialVectors = "data/mario/initialVectors.txt";
    static String logPath = "data/mario/";
    
    static int filterWidth = 2;
    static int filterHeight = 2;
    static int stride = 1;
    
    
    public static void main(String[] args) throws Exception {
        String filename;
        filename = "KLDivLog-fw"+filterWidth+"-fH"+filterHeight+"-s"+stride+".txt";
        FileWriter write = new FileWriter(logPath+filename , true);
        PrintWriter print_line = new PrintWriter( write );
        
        //Build ConvNTuples for training Levels        
        File folderTrain = new File(trainingInputPath);
        File[] listOfFilesTrain = folderTrain.listFiles();
        ArrayList<int[][]> trainingLevels = new ArrayList<>(listOfFilesTrain.length);
        ArrayList<ConvNTuple> trainingLevelsConv = new ArrayList<>(listOfFilesTrain.length);
        ConvNTuple trainingLevelsAllConv = new ConvNTuple();
        for (int i=0; i< listOfFilesTrain.length; i++){
            String inputFile = listOfFilesTrain[i].getAbsolutePath(); 
            int[][] level = readLevel(new Scanner(new FileInputStream(inputFile)), tiles);
            trainingLevels.add(i, level);
            ConvNTuple c = getConvNTuple(level, filterWidth, filterHeight, stride);
            trainingLevelsConv.add(i, c);
            if(i==0){
                trainingLevelsAllConv = getConvNTuple(level, filterWidth, filterHeight, stride);
            }else{
                trainingLevelsAllConv.sampleDis.add(c.sampleDis);
            }
        }
        
        //Build ConvNTuples for initial vectors
        File listOfInitialVectors = new File(initialVectors);
        ArrayList<String> linesIni = new ArrayList<>();
        Scanner scannerIni = new Scanner(new FileInputStream(listOfInitialVectors));
        while (scannerIni.hasNext()) {
            String line = scannerIni.nextLine();
            linesIni.add(line);
        }
        ArrayList<int[][]> initLevels = new ArrayList<>(linesIni.size());
        ArrayList<ConvNTuple> initLevelsConv = new ArrayList<>(linesIni.size());
        ConvNTuple initLevelsAllConv = new ConvNTuple();
        for (int i=0; i<linesIni.size(); i++){
            String strLatentVector = linesIni.get(i);
            strLatentVector = strLatentVector.replace("[", "");
            strLatentVector = strLatentVector.replace("]", "");
            strLatentVector = strLatentVector.replace(" ", "");
            String[] parts = strLatentVector.split(",");
            double [] level = new double[parts.length];
            for(int j=0; j<parts.length; j++){
                level[j] = Double.valueOf(parts[j]);
            }
            Parameters.initializeParameterCollections(new String[] {"marioGANModel:Mario1_Overworld_5_Epoch5000.pth","GANInputSize:5", "marioGANUsesOriginalEncoding:false"});
            ArrayList<List<Integer>> levelIntRep =  MarioGANUtil.generateLevelListRepresentationFromGAN(level);
            int[][] levelArray = levelIntRep.stream()                                // Stream<List<Integer>>
                .map(list -> list.stream().mapToInt(k -> k).toArray()) // Stream<int[]>
                .toArray(int[][]::new);
            
            initLevels.add(i, levelArray);
            ConvNTuple c = getConvNTuple(levelArray, filterWidth, filterHeight, stride);
            initLevelsConv.add(i, c);
            if(i==0){
                initLevelsAllConv = getConvNTuple(levelArray, filterWidth, filterHeight, stride);
            }else{
                initLevelsAllConv.sampleDis.add(c.sampleDis);
            }
        }
        
        
        

        //Build ConvNTuples for interactive generated data - only last generation
        File folderInt = new File(interactiveInputPath);
        File[] listOfRuns = folderInt.listFiles();
        ArrayList<int[][]> intLevels = new ArrayList<>();
        ArrayList<ConvNTuple> intLevelsConv = new ArrayList<>();
        ConvNTuple intLevelsAll = new ConvNTuple(); //(interactive - all experiments)
        ArrayList<ConvNTuple>  intLevelsExp = new ArrayList<>(); //(interactive - per user)
        ArrayList<ConvNTuple>  intLevelsType = new ArrayList<>(); //(interactive - per type)
        //ArrayList<ConvNTuple>  intLevelsExpType = new ArrayList<>(); //(interactive - per user - per type)
       
        
        FilenameFilter genFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                String lowercaseName = name.toLowerCase();
                if (lowercaseName.startsWith("gen")) {
                    return true;
                } else {
                    return false;
                }
            }
        };   
        FilenameFilter vectorFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                String lowercaseName = name.toLowerCase();
                if (lowercaseName.startsWith("vector")) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        
        FilenameFilter expFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                String lowercaseName = name.toLowerCase();
                if (lowercaseName.startsWith("both") || lowercaseName.startsWith("evolve") ) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        
        
        int id = 0;
        for (int run=0; run< listOfRuns.length; run++){
            File folderIntExp = new File(listOfRuns[run].getAbsolutePath());
            File[] listOfTypes = folderIntExp.listFiles(expFilter);
            for(int type=0; type<listOfTypes.length; type++){
                File folderIntExpGens = new File(listOfTypes[type].getAbsolutePath());               
                File[] listOfGens = folderIntExpGens.listFiles(genFilter);
                for(int gen=0; gen < listOfGens.length; gen++){
                    File folderIntExpGensResults = new File(listOfGens[gen].getAbsolutePath());
                    int generation = Integer.parseInt(folderIntExpGensResults.getName().replaceAll("\\D+",""));
                    File[] listOfVectors = folderIntExpGensResults.listFiles(vectorFilter);
                    for(int vec=0; vec < listOfVectors.length; vec++){
                        File levelFile = new File(listOfVectors[vec].getAbsolutePath());
                        int vector = Integer.parseInt(levelFile.getName().replaceAll("\\D+",""));
                        print_line.println(id + "\t" + run + "\t" + type + "\t" + generation + "\t" + vector + "\t" + levelFile.getPath());
                        ArrayList<String> lines = new ArrayList<>();
                        Scanner scanner = new Scanner(new FileInputStream(levelFile));
                        while (scanner.hasNext()) {
                            String line = scanner.nextLine();
                            lines.add(line);
                        }
                        String strLatentVector = lines.get(0);
                        strLatentVector = strLatentVector.replace("[", "");
                        strLatentVector = strLatentVector.replace("]", "");
                        strLatentVector = strLatentVector.replace(" ", "");
                        String[] parts = strLatentVector.split(",");
                        double [] level = new double[parts.length];
                        for(int i=0; i<parts.length; i++){
                            level[i] = Double.valueOf(parts[i]);
                        }
                        Parameters.initializeParameterCollections(new String[] {"marioGANModel:Mario1_Overworld_5_Epoch5000.pth","GANInputSize:5", "marioGANUsesOriginalEncoding:false"});
                        ArrayList<List<Integer>> levelIntRep =  MarioGANUtil.generateLevelListRepresentationFromGAN(level);
                        int[][] levelArray = levelIntRep.stream()                                // Stream<List<Integer>>
                            .map(list -> list.stream().mapToInt(i -> i).toArray()) // Stream<int[]>
                            .toArray(int[][]::new);
                        intLevels.add(id, levelArray);
                        ConvNTuple c = getConvNTuple(levelArray, filterWidth, filterHeight, stride);
                        intLevelsConv.add(id, c);
                        if(id==0){
                            intLevelsAll = getConvNTuple(levelArray, filterWidth, filterHeight, stride);
                        }else{
                            intLevelsAll.sampleDis.add(c.sampleDis);
                        }
                        if(intLevelsExp.size()<=run){
                            intLevelsExp.add(run,getConvNTuple(levelArray, filterWidth, filterHeight, stride));
                        }else{
                            ConvNTuple tmp = intLevelsExp.get(run);
                            tmp.sampleDis.add(c.sampleDis);
                            intLevelsExp.set(run, tmp);
                        }
                        if(intLevelsType.size()<=type){
                            intLevelsType.add(type,getConvNTuple(levelArray, filterWidth, filterHeight, stride));
                        }else{
                            ConvNTuple tmp = intLevelsType.get(type);
                            tmp.sampleDis.add(c.sampleDis);
                            intLevelsType.set(type, tmp);
                        }                        
                        id++;
                    }
                }
            }
        }

        print_line.println("1 KL-Div to general distribution of training levels (overworld)");
        print_line.println("1-1 from single training levels");
        for(int i=0; i<trainingLevelsConv.size(); i++){
            print_line.println("1-1\t"+KLDiv.klDiv(trainingLevelsAllConv.sampleDis, trainingLevelsConv.get(i).sampleDis));
        }
        print_line.println("1-2 from single generated levels (interactive)");
        for(int i=0; i<intLevelsConv.size(); i++){
            print_line.println("1-2\t"+KLDiv.klDiv(trainingLevelsAllConv.sampleDis, intLevelsConv.get(i).sampleDis));
        }
        print_line.println("1-3 from random generated levels");
        for(int i=0; i<initLevelsConv.size(); i++){
            print_line.println("1-3\t"+KLDiv.klDiv(trainingLevelsAllConv.sampleDis, initLevelsConv.get(i).sampleDis));
        }
        print_line.println("1-4 from general distribution of generated levels (interactive - per user)");
        for(int i=0; i<intLevelsExp.size(); i++){
            print_line.println("1-4\t"+KLDiv.klDiv(trainingLevelsAllConv.sampleDis, intLevelsExp.get(i).sampleDis));
        }
        print_line.println("1-5 from general distribution of generated levels (interactive - per type)");
        for(int i=0; i<intLevelsType.size(); i++){
            print_line.println("1-5\t"+KLDiv.klDiv(trainingLevelsAllConv.sampleDis, intLevelsType.get(i).sampleDis));
        }
        print_line.println("1-6 from general distribution of generated levels (interactive - per user - per type)");
        print_line.println("1-7 from general distribution of generated levels (interactive - all experiments)");
        print_line.println("1-7\t"+KLDiv.klDiv(trainingLevelsAllConv.sampleDis, intLevelsAll.sampleDis));
        print_line.println("1-8 from general distribution of randomly generated levels");
        print_line.println("1-8\t"+KLDiv.klDiv(trainingLevelsAllConv.sampleDis, initLevelsAllConv.sampleDis));
        
        print_line.println("2 KL-Div between single levels (distance matrix)");
        for(int i=0; i<trainingLevelsConv.size(); i++){
            for(int j=0; j<trainingLevelsConv.size(); j++){
                print_line.println("2\tt"+i+"\tt"+j+"\t"+KLDiv.klDiv(trainingLevelsConv.get(i).sampleDis, trainingLevelsConv.get(j).sampleDis));
            }
            for(int j=0; j<intLevelsConv.size(); j++){
                print_line.println("2\tt"+i+"\ti"+j+"\t"+KLDiv.klDiv(trainingLevelsConv.get(i).sampleDis, intLevelsConv.get(j).sampleDis));
            }
        }
        for(int i=0; i<intLevelsConv.size(); i++){
            for(int j=0; j<trainingLevelsConv.size(); j++){
                print_line.println("2\ti"+i+"\tt"+j+"\t"+KLDiv.klDiv(intLevelsConv.get(i).sampleDis, trainingLevelsConv.get(j).sampleDis));
            }
            for(int j=0; j<intLevelsConv.size(); j++){
                print_line.println("2\ti"+i+"\ti"+j+"\t"+KLDiv.klDiv(intLevelsConv.get(i).sampleDis, intLevelsConv.get(j).sampleDis));
            }
        }
        
        
        
        print_line.close();
        write.close();
        
        
        
        //ArrayList<ConvNTuple> trainingLevelsConv = new ArrayList<>(listOfRuns.length);
        //ConvNTuple trainingLevelsAllConv = new ConvNTuple();

        
        //for different test settings of stride and window size etc, test:
            //KL-Div to general distribution of training levels (overworld)
                //from single training levels
                //from single generated levels (interative)
                //from random generated levels
                //from general distribution of generated levels (interactive - per user)
                //from general distribution of generated levels (interactive - per type)
                //from general distribution of generated levels (interactive - per user - per type)
                //from general distribution of generated levels (interactive - all experiments)
                //from general distribution of randomly generated levels
            
            //KL-Div between single levels (distance matrix)
                //training levels
                //single generated levels (interactive)

        
        
    }

}
