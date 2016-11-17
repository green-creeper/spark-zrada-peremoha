# spark-zrada-peremoha
Text classification with spark-mllib

There is social phenomenon in Ukraine. Any news that appear are got label either win or betray (зрада or перемога).
I executed script for twitter which collects messages with appropriate hashtags.
After cleanup and extracting I made a CSV file which you could find in repository.
There are two fields text and isWin

Spark pipeline looks like this:

Tokenizer -> StopWordsCleaner -> HashTF to extract features -> NaiveBayes classifier.

I experimented with bunch of classifier and also executed Cross Validation, but NaiveBayes gives the best result.
