Dataset(Sejong corpus) can be downloaded at:  
https://drive.google.com/file/d/1dSfw9F2-XHMz6Zv6NpIZEYa50L0kRpBe/view?usp=sharing

Dataset provided is processed so that every line
has approximately 200 characters to fit the default model configuration. 

## Quick notes on the dataset
When creating a custom dataset, concatenating shorter sentences into a single line can help 
reduce training time without hurting the performance.  

50,000 lines were randomly selected from the corpus(152 characters per line) for training, 
and the same lines were stretched to 100,000(76 characters per line) for comparison.  

![](../images/20191211_comapre_preprocessing_performance.png)

Two models with the default configuration were trained on each dataset, and the former showed comparable 
performance on separate validation set after 10 epochs as the latter, with just half the training time.  

## Credits
Thanks to [EthanJYK](https://github.com/EthanJYK) for processing the dataset. 
