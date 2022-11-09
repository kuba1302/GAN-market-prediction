# Stock price prediction using GAN and BERT
### Description of the project
In this project the use of the Generative adversarial networks (GAN) model to predict
stock market behavior has been proposed. Additionally, an investment strategy that uses
predictions from the model has been created.
The Technical Analysis and Sentiment Analysis indicators are used as explanatory variables for the model. The study was conducted on four companies from the game industry - Electronic Arts, Ubisoft, TakeTwo Interactive Software and Activision Blizzard. This industry was chosen because of the potential impact of a company’s customer reviews on its valuation. The impact
is investigated by analyzing the sentiment of selected comments containing prepared
keywords. The sentiment is calculated using created by Google NLP model - BERT.
GAN tries to predict the future valuation of a given company based on the aforementioned
predictors. The price forecast is then used to provide buy or sell signals. In the study, for
3 out of 4 companies it was possible to create a strategy that significantly outperforms
the Buy and Hold approach. The main finding of the research is the confirmation of the
possibility to use GAN networks to create an effective system that allows to outperform
the Buy and Hold strategy and make a profit despite the fall in the price of an asset.

### Most important files
- Implementation of GAN is located [here](https://github.com/kuba1302/GAN-market-prediction/blob/main/models/gan/gan.py) 
- Paper is located [here](https://github.com/kuba1302/GAN-market-prediction/blob/main/paper/gan_stock.pdf)
- Data preprocessing is located [aere](https://github.com/kuba1302/GAN-market-prediction/blob/main/data_pipelines/data_preprocessing.py)
