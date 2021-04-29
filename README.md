# Moral_Foundation_FrameAxis
This library provides the code for a flexible calculation of moral foundation scores from textual input data.
We have developed this framework inspired by [FrameAxis paper](https://arxiv.org/pdf/2002.08608.pdf).

Using this framework we have studied two applications:
- [Moral Framing and Ideological Bias of News, 2020](https://arxiv.org/pdf/2009.12979.pdf)
- [Mapping Moral Valence of Tweets Following the Killing of George Floyd, 2021](https://arxiv.org/pdf/2104.09578.pdf)

If you are using this code please consider citing our papers and giving this repository a star.

### Options for Dictionaries:
FrameAxis is a flexible framework and all it needs is sets of antonym words. There are several moral foundation dictionaries that provide vice and virtues of moral categories. In this code we are supporting:
- **MFD:**
- **MFD2.0**: 
- **eMFD:** A moral foundations dictionary extracted from human-annotated corpra by [Hopp et al.](https://link.springer.com/article/10.3758/s13428-020-01433-0).
- **customized**: If you don't want to stick to these dictionaries; maybe you want to curate your own dictionary in your language or go out of the scope of moral foundations you can choose the dictionary argument as "--dict_type custom" and provide your file in moral_foundations_dictionary folder. Your dictionary file must be named as "custom.csv" and must have three columns named as "word", "category", and "polarization". Same as the MFD.csv which is already provided. 
## Command-Line Arguments
- **[--input_file]:** Path to the dataset .csv file containing input text documents in a column.
- **[--docs_colname]:** The name of the column in the input file that contains the texts to calculate the MF scores on. 

- **[--dict_type]:** Dictionary for calculating FrameAxis Scores. Possible values are: emfd, mfd, mfd2, and custom. 
- **[--word_embedding_model]:** Path to the word embedding model used to map words to a vector space. If not specified a default w2v model will be used.
- **[--output_file]:** The path for saving the MF scored output CSV file. The output file contains columns for MF scores concatenated to the original dataset.

### To-dos: 
- word-embedding updating tool + downloading the default
- Supporting MFD2.
- Document TFIDF tool. 
- Add package requirements and documentation for how to setup python. 
