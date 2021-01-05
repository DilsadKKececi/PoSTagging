# @author: Dilsad Ergun Kucukkececi

import math
import time
import string
import csv


LOG_OF_ZERO_FOR_VITERBI=-1000 #for viterbi table, explained in report in details

"""
    Method for calculating counts of tag bigrams
    """

def transitionCounts(trainData):

    tagTagPairs={}
    singleLine=[]
    allLines=[]
    iterator=0
    end='End'
    
    for sentence in trainData:
        line=sentence.split(' ') #find each pair
        for pair in line:
            tag=pair.split('/')[1]  #get tag
            if tag.endswith('\n'):
                tag=tag.replace('\n','')
            singleLine.append(tag) #add tag of sentence to location
        allLines.append(singleLine) #add sentence with only tags
        singleLine=[]

    tagTagPairs['Start']={}


    for sentence in allLines:
        
        #add first tag as initial probability
        
        if sentence[iterator] in tagTagPairs['Start']: #if added before
            tagTagPairs['Start'][sentence[iterator]]+=1
        else:
             tagTagPairs['Start'][sentence[iterator]]=1 #if seen new

        #process for tags after first
        
        while iterator<len(sentence)-1:
            
            if sentence[iterator] in tagTagPairs: #if added before
                if sentence[iterator+1] in tagTagPairs[sentence[iterator]]: #if bigram seen before
                    tagTagPairs[sentence[iterator]][sentence[iterator+1]]+=1 #increase count by one
                else:
                    tagTagPairs[sentence[iterator]][sentence[iterator+1]]=1 #initialize count
            else:
                tagTagPairs[sentence[iterator]]={} #if new tag create empty dictionary
                tagTagPairs[sentence[iterator]][sentence[iterator+1]]=1 #initialize count

            if iterator==len(sentence)-2:
                
                if sentence[iterator+1] in tagTagPairs: #last tag of word exists
                    
                    if end in tagTagPairs[sentence[iterator+1]]: #if end token came after that before
                        
                        tagTagPairs[sentence[iterator+1]][end]+=1 #increase count by one
                    else: tagTagPairs[sentence[iterator+1]][end]=1 #if end token did not come after that before initialize count

                else: #last tag of word does not exist
                    tagTagPairs[sentence[iterator+1]]={}
                    tagTagPairs[sentence[iterator+1]][end]=1
            iterator+=1
                        
            
        iterator=0

    return tagTagPairs #counts of bigrams

"""
    Method for calculating probabilities of tag bigrams
    """

def transitionProbabilities(transitionCounts):

    totalCount=0
    count=0
    for eachKey in transitionCounts:
        totalCount=sum(transitionCounts[eachKey].values()) #sum of all counts of the related tag
        for innerKey in transitionCounts[eachKey].keys():
            count=transitionCounts[eachKey][innerKey] #count value of bigram
            if count ==0:
                transitionCounts[eachKey][innerKey]=LOG_OF_ZERO_FOR_VITERBI #if bigram does not exists
            else:
            
                transitionCounts[eachKey][innerKey]=math.log2(count/totalCount) #if bigram exist calculate lof of probability


    transitionCounts['End']={}
    for i in transitionCounts.keys():
        transitionCounts['End'][i]=LOG_OF_ZERO_FOR_VITERBI #no tag can come after end token

    keysOf=list(transitionCounts.keys())
    i=0
    while i<len(keysOf):
        for j in transitionCounts.keys():
            
            if j not in transitionCounts[keysOf[i]].keys():
                transitionCounts[keysOf[i]][j]=LOG_OF_ZERO_FOR_VITERBI #add all tags under each tag's dictionary if does not exist
        i+=1
    for z in transitionCounts:
        if 'End' not in transitionCounts[z].keys():
            transitionCounts[z]['End']=LOG_OF_ZERO_FOR_VITERBI #no tag can come after end token
    

    return transitionCounts #probabilities of bigrams

"""
    Method for calculating counts of tag word-tag pairs
    """

def emissionCounts(trainData):
    
    
    wordTagPairs={}
    
    
    for sentence in trainData:
        line=sentence.split(' ') #reach pairs
        
        for pair in line:
            
            word=(pair.split('/')[0]).lower() #get word, turn to lowercase

            tag=pair.split('/')[1] #get tag
            
            if tag in wordTagPairs: #if tag is seen before
                if word in wordTagPairs[tag]: #if word is seen before under related tag
                    wordTagPairs[tag][word]+=1 #increase count
                else:
                    wordTagPairs[tag][word]=1 #initialize count
            
            
            else: #if non seen tag add and initialize
                wordTagPairs[tag]={}
                wordTagPairs[tag][word]=1

    #none of the words is either start or end

    wordTagPairs['Start']={}
    wordTagPairs['End']={}
    

    return wordTagPairs

"""
    Method for calculating probabilities of tag word-tag pairs
    """

def observationLikeliHoods(emissionCounts):

    totalCount=0
    count=0
    for eachKey in emissionCounts:
        totalCount=sum(emissionCounts[eachKey].values()) #sum of all values of related tag
        
        for innerKey in emissionCounts[eachKey].keys():
            count=emissionCounts[eachKey][innerKey] #count of word under related tag
            if count==0:
                emissionCounts[eachKey][innerKey]=LOG_OF_ZERO_FOR_VITERBI
            else:
            
            
                emissionCounts[eachKey][innerKey]=math.log2(count/totalCount) #calculate log of probability

    for eachKey in emissionCounts.keys():
        if eachKey=='End':
        
            emissionCounts[eachKey]['End_Of_Sentence']=0 #define end of sentence possibility with probability of 1 so log is 0


    #add all punctuations with probability of 1 so log is 0
    for i in string.punctuation:
        emissionCounts['Punc'][i]=0
        emissionCounts['Punc'][i+i]=0


    return emissionCounts



"""
    Viterbi algorithm
    """
def viterbi(obs, states, start_p, trans_p, emit_p, smooth):
    
    obs.append('End_Of_Sentence') #add symbol to end
    constLength=len(obs)
    
    V=[{}] #initialize viterbi table
    backTraces=[{}] #initialize bac trace table
    
    #use initial possibilities at the beginning
    
    for i in states:
        
        if obs[0] in emit_p[i]: #if word seen before
        
            V[0][i]=start_p[i]+emit_p[i][obs[0]] #assign probability
            backTraces[0][i]='Start' #give backtrace as Start tag

        else:
            V[0][i]=start_p[i]+LOG_OF_ZERO_FOR_VITERBI #assign probability
            backTraces[0][i]='Start' #give backtrace as Start tag


    # Run Viterbi when t > 0
    for t in range(1, constLength):
        
        V.append({}) #initialize new dictionary for next word
        backTraces.append({}) #initialize new dictionary for next word's backtrace
        
        
        for y in states:
            x=LOG_OF_ZERO_FOR_VITERBI #initial value
            
            if obs[t] in emit_p[y]:
                
                x=emit_p[y][obs[t]] #if exists get probability from observation likelihoods
            
            elif(y=='Noun'): #special case for nearest neigbour
                x=smooth #give minimum probability of neighbour



            (prob, state) = max((V[t-1][prev] + trans_p[prev][y] + x, prev) for prev in states  )
            
            #calculate probability as maximum with considering the previous cell
            #assign state as which probability was the maximum
            #state is where I came from, the back trace pointer will show that
            
            #fill structures with values
            
            V[t][y] = prob
            backTraces[t][y]=state

        
        
        
    test=[] #will hold path

    endColumn=V[constLength-1]
    

    for x,y in V[constLength-1].items():
        
        if endColumn[x]==max(endColumn.values()): #find max of last column
            test.append(x)

    last=constLength-1
    while last>=0:
        for j in V[last].keys():
            
            if V[last][j]==max(V[last].values()):
                
                test.append(backTraces[last][j]) #follow backtraces
        last-=1
    
    test.reverse() #reverse the path since we came from back

    
    return(test) #return path


"""
    Find Correct tags to use in evaluation
    """

def getRightTags(testData):

    correctTags=[]
    singleLine=[]
    for sentence in testData:
        line=sentence.split(' ')
        singleLine.append('Start') #add start symbol
        for pair in line:
            tag=pair.split('/')[1] #get tags from sentence
            if tag.endswith('\n'):
                tag=tag.replace('\n','')
            singleLine.append(tag)
        singleLine.append('End') #add end symbol
        
        correctTags.append(singleLine) #append tags of sentence
        singleLine=[]
    return correctTags

"""
    Evaluation process
    """

def evaluateResult(transitionProbs, emissionProbs, testData):
    
    startPossibilities=transitionProbs['Start'] #find initial possibilities
    endPossibilities={}
    
    for tag in transitionProbs.keys():
        
        endPossibilities[tag]=transitionProbs[tag]['End'] #find end possibilities
    
    states=tuple(transitionProbs.keys()) #find all tags

    correctTags=getRightTags(testData) #find correct tags

    iterator=0
    total=0
    totalWords=0
    
    smooth=min(emissionProbs['Noun'].values()) #calculate smoothing value
    
    
    
    
    for sentence in testData:
        
        line=sentence.split(' ')
        line = [element.lower() for element in line] #turn all to lower
        line = [element.split('/')[0] for element in line] #only get words, not tags
        
        
        result=viterbi(line, states, startPossibilities, transitionProbs, emissionProbs, smooth) #call viterbi for each sentence
        
        
        
        count=0
        
        for i in range(0,len(result)):
            if result[i]==correctTags[iterator][i]: #compare result with the correct
                count+=1 #if word is matched
    

        totalWords+=len(result)
        total+=count #add to total
        iterator+=1

    return((total/totalWords)*100)




"""
    Creates CSV for Kaggle, since CSV does not contain Start, End tags in evaluation probability becomes %84,32, comparison file is in the folder data_test_words.csv
    """

def createCSVForKaggle(transitionProbs, emissionProbs, testData):
    
    csvData = []
    csvFollow=0
    csvData.append(['Id,Category'])
    
    startPossibilities=transitionProbs['Start'] #find initial possibilities
    endPossibilities={}
    
    for tag in transitionProbs.keys():
        
        endPossibilities[tag]=transitionProbs[tag]['End'] #find end possibilities
    
    states=tuple(transitionProbs.keys()) #find all tags

    correctTags=getRightTags(testData) #find correct tags

    iterator=0
    total=0
    totalWords=0
    
    smooth=min(emissionProbs['Noun'].values()) #calculate smoothing value
    
    
    
    
    for sentence in testData:
        
        line=sentence.split(' ')
        line = [element.lower() for element in line] #turn all to lower
        line = [element.split('/')[0] for element in line] #only get words, not tags
        
        
        result=viterbi(line, states, startPossibilities, transitionProbs, emissionProbs, smooth)
        #call viterbi for each sentence
        tagSequence=1
        while tagSequence <len(result)-1:
            csvFollow+=1
            row=[str(csvFollow)+','+result[tagSequence]]
            
            csvData.append(row)
            tagSequence+=1
        
        
        
        
        count=0



    with open('output.csv', 'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=';', quotechar="", quoting=csv.QUOTE_NONE)
        writer.writerows(csvData)
    
    csvFile.close()
    
    counterForDiff=0
    
    
    
    with open('output.csv', 'r') as t1, open('data_test_words.csv', 'r') as t2:
        fileone = t1.readlines()
        filetwo = t2.readlines()
    
    for line in filetwo:
        if line not in fileone:
            counterForDiff+=1


    print(str(((18154-counterForDiff)/18154)*100)+'%')


"""
    Can be used for found taggings with comparisons
    """
def displayTaggings(transitionProbs, emissionProbs, testData):
    
    foundTagSequence=[]
    realTagSequence=[]
    
    startPossibilities=transitionProbs['Start'] #find initial possibilities
    endPossibilities={}
    
    for tag in transitionProbs.keys():
        
        endPossibilities[tag]=transitionProbs[tag]['End'] #find end possibilities
    
    states=tuple(transitionProbs.keys()) #find all tags

    correctTags=getRightTags(testData) #find correct tags

    iterator=0
    total=0
    totalWords=0
    
    smooth=min(emissionProbs['Noun'].values()) #calculate smoothing value
    
    
    
    
    for sentence in testData:
        
        realTagSequence.append(sentence)
        
        
        line=sentence.split(' ')
        line = [element.split('/')[0] for element in line] #only get words, not tags
        beforePreProcess=line
        line = [element.lower() for element in line] #turn all to lower
        
        
        
        result=viterbi(line, states, startPossibilities, transitionProbs, emissionProbs, smooth) #call viterbi for each sentence
        
        startFrom=1
        follower=0
        sequenceFoundForSentence=[]
        
        while startFrom<len(result)-1:
            
            sequenceFoundForSentence.append(beforePreProcess[follower] + '/'+ result[startFrom])
            
            follower+=1
            startFrom+=1
        
        foundTagSequence.append(sequenceFoundForSentence)
        

        
        count=0
        
        for i in range(0,len(result)):
            if result[i]==correctTags[iterator][i]: #compare result with the correct
                count+=1 #if word is matched
    
        
        totalWords+=len(result)
        total+=count #add to total
        iterator+=1
        
        
        i=0
        while i<len(foundTagSequence):
            print("")
            print(realTagSequence[i])
            print(foundTagSequence[i])
            i+=1

if __name__ == "__main__":  # main program

    trainCounter=0
    
    #read file's 70 percentage as train, 30 percentage as test

    with open('data.txt','r',encoding='utf8',errors="ignore") as allData:
        trainData=[]
        testData=[]
        for line in allData:
            if(trainCounter<3960):
                trainData.append(line)
            else: testData.append(line)
            trainCounter+=1


    # create hidden markov model elements

    #calculate counts
    transitionProbs=(transitionCounts(trainData))
    emissionProbs=(emissionCounts(trainData))

    #calculate probabilities
    transitionProbs=transitionProbabilities(transitionProbs)
    emissionProbs=observationLikeliHoods(emissionProbs)

    #evaluate result
    percentage=evaluateResult(transitionProbs, emissionProbs, testData)
    print("Match percentage is: %" + str(percentage))

    #createCSVForKaggle(transitionProbs, emissionProbs, testData)
    #displayTaggings(transitionProbs, emissionProbs, testData)
    







