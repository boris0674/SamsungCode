## i pacchetti utilizzati


library("dplyr")
library(forecast)
library(moments)
library(ggplot2)
library(iprior)
library(lubridate)
library(gridExtra)
library(aTSA)
library(tsoutliers)
library(tseries)
library(nortest)
library(rugarch)
library(keras)
library(ElemStatLearn)
library(caret)
library(rpart)
library(e1071)
library(parallel)
library(doParallel)
library(LaplacesDemon)
library(aTSA)
library(tsoutliers)
library(fitdistrplus)
library(PerformanceAnalytics)
library(quantmod)


## importiamo i dati e creiamo log-rendimenti di Samsung Electronics

options(scipen=999)
samsung <- read.csv("samsung.csv")
samsung <- samsung$Close
samsung <- rev(samsung)
samsung_yld <- log(samsung[1:2614]/samsung[2:2615])
samsung_yld <- rev(samsung_yld)
training <- samsung_yld[1:1743]
test <- samsung_yld[1744:2614]

## dati per chart orezzi Samsung
samprice <- read.csv("samsungtimeseries.csv")
samprice <- samprice$Close
samdate <- read.csv("samdate.csv")
samdate <- samdate$Date
samdate <- as.Date(samdate)
sams <- data.frame(samdate, samprice)
ggplot(sams, aes(samdate, samprice)) + geom_line(color="steelblue") +
  labs(x= "l'andamento di Samsung Electronics da fine 2008", y=" Prezzo") 

## analisi ARIMA
adf.test(training)
auto.arima(training)
acf(training)
pacf(training)
Box.test(training, lag=45, type="Ljung-Box")
trainingfit <- arima(training, order=c(2,0,1))
resid_tr <- resid(trainingfit, standardize=T)
resid_tr2 <- resid(trainingfit, standardize=F)
tsdisplay(resid_tr)
Box.test(resid_tr, lag=45, type="Ljung-Box")
arima_forecast <- forecast(trainingfit, 871)
mean_train <- rep(0.0056, 871)
resid_ts <- test-mean_train
cor(fitted(trainingfit), training)^2
mae_train <- sum(abs(resid_tr2))/1739
mae_test <- sum(abs(resid_ts))/867


## rete neurale per individuare dipendenze a lungo termine
samlag <- Lag(samsung_yld, k=1:45)
samlag <- data.frame(samlag)
samlag$Lag.0 <- samsung_yld
samlag <- samlag[complete.cases(samlag),]
train <- samlag[1:1743,]
test <- samlag[1744:2547, ]
x_train2 <- subset(train, select=-c(Lag.0))
y_train2 <- subset(yentrain, select=c(Lag.0))
x_test2  <-  subset(test, select=-c(Lag.0))  
y_test2 <-  subset(test, select=c(Lag.0))   
x_train3 <- as.matrix(x_train2)
x_test3 <- as.matrix(x_test2)
y_train3 <- as.matrix(y_train2)
y_test3 <- as.matrix(y_test2)


## il codice per interfacciarsi con TensorFlow
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 32, kernel_initializer="random_uniform", bias_initializer="zeros",
              activation = "linear", input_shape = c(45),
              kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01) ) %>% 
  layer_dropout(rate = 0.8) %>% 
  layer_dense(units = 16, activation = "linear", kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01) ) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 8, activation = "linear", kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01) )  %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 4, activation = "linear", kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01) ) %>%
  
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 2, activation = "linear", kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01) ) %>%
  
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

model %>% compile(
  loss = "mse",
  optimizer = "sgd",
  metrics = "mae"
)

history <- model %>% fit(
  x_train3, y_train3, 
  epochs = 150, batch_size = 50, 
  validation_split = 0.2
)

val_pre <- model %>% evaluate(x_test3, y_test3)

plot(history)


## le caratteristiche stocastiche di Samsung Electronics

shapiro.test(training)
jarque.bera.test(training)
ad.test(training)
min(training)
max(training)
mean(training)
median(training)
sd(training)
skewness(training)
kurtosis(training)

min(test)
max(test)
mean(test)
median(test)
sd(test)
skewness(test)
kurtosis(test)

labs2 <- (abs(training-median(training)))
labs3 <- (abs(test-median(test)))
sum(labs2)/1743
sum(labs3)/871
set.seed(50)
laplace2 <- rlaplace(1743, 0, s=0.013851)
laplace3 <- rlaplace(871, 0, s=0.013851)
laplace4 <- rlaplace(871, 0.001115449, s=0.01215253)
ks.test(training, laplace2)
ks.test(test, laplace3)
ks.test(test, laplace4)
df_laplace2 <- data.frame(laplace2)
df_laplace3 <- data.frame(laplace3)
df_training <- as.data.frame(training)
df_test <- as.data.frame(test)


ggplot(df_training, aes(x=training),colour="green" )+
  stat_density( kernel="gaussian", n=512, bw="nrd0",fill="darkred"  )+
  geom_density( aes(df_laplace2$laplace2))+
  ggtitle("Stima Kernel della distribuzione yld di samsung-tr+campione Laplace ")+
  ylab ("densità di probabilità")+ xlim(min(df_training$training),max(df_training$training))+
  geom_vline(xintercept=mean(df_training$training))

ggplot(df_training, aes(x=training),colour="green" )+
  stat_density( kernel="gaussian", n=512, bw="nrd0",fill="darkred"  )+
  geom_density( aes(df_laplace2$laplace2))+
  ggtitle("Stima Kernel della distribuzione yld di samsung-tr+campione Laplace ")+
  ylab ("densità di probabilità")+ xlim(min(df_training$training),max(df_training$training))+
  geom_vline(xintercept=mean(df_training$training))


## VAEr e ETL di Samsung Electronics

varnor <- VaR(training, p=0.99, method=  "gaussian")
varcorn <- VaR(training, p=0.99, method=  "modified")

tibs <- tibble(num = 1:1000) %>% 
  group_by(num) %>% 
  mutate(vars = quantile(sample(training, 1000000, replace = TRUE), p=0.01))
tibs2 <- data.frame(tibs)
varmean <-  mean(tibs2$vars)

garchOrder <- c(1,1) 
varModel <- list(model = "sGARCH", garchOrder = garchOrder)
spec <- ugarchspec(variance.model= list(model = "sGARCH", garchOrder = c(1,1) )
                   , mean.model= list(armaOrder=c(0,0)), distribution.model="std")
fit_garch <- ugarchfit(spec, data=training)
coef(fit_garch)
trainsq <- training^2
ratio <- trainsq/variances
Box.test(trainsq, lag=45, type="Ljung-Box")
Box.test(ratio, lag=45, type="Ljung-Box")
garch_var <- quantile(fit_garch, 0.01)
min(garch_var)
max(garch_var)
spec2 <- as.list(coef(fit_garch))
spec3  <- ugarchspec(variance.model= list(model = "sGARCH", garchOrder = c(1,1) )
                     , mean.model= list(armaOrder=c(0,0)), distribution.model="std", fixed.pars = spec2)
garch_fcast <- ugarchforecast(spec3, n.ahead = 871, data=samsung_yld, n.roll=1, out.sample=871)
fitted_fgarch <- fitted(garch_fcast)
fextr <- quantile(garch_fcast, p=0.01)

length(training[training <= varnor])
length(training[training <= varcorn])
length(training[training <= varmean])
length(training[training+abs(garch_var) <= 0])

length(test[test <= varnor])
length(test[test <= varcorn])
length(test[test<= varmean])
length(test[test+abs(fextr[1:871]) <=0])


gaussian2 <- rep(varnor, times=1743)
modified2 <- rep(varcorn, times=1743)
historical2 <- rep(varmean, times=1743)
gaussian <- rep(varnor, times=871)
modified <- rep(varcorn, times=871)
historical <- rep(varmean, times=871)
VaRTest(alpha=0.01, training, VaR=gaussian2, conf.level=0.95)
VaRTest(alpha=0.01, training, VaR=modified2, conf.level=0.95)
VaRTest(alpha=0.01, training, VaR=historical2, conf.level=0.95)
VaRTest(alpha=0.01, training, VaR=garch_var[1:1743], conf.level=0.95)
VaRTest(alpha=0.01, test, VaR=gaussian, conf.level=0.95)
VaRTest(alpha=0.01, test, VaR=modified, conf.level=0.95)
VaRTest(alpha=0.01, test, VaR=historical, conf.level=0.95)
VaRTest(alpha=0.01, test, VaR=fextr[1:871], conf.level=0.95)

ES(training, p=0.99, method="gaussian")
ES(training, p=0.99, method="modified")
mean(training[training <=quantile(training, p=0.01)])
mean(test[test <=quantile(test, p=0.01)])


## regressions

samsung2<- read.csv("samsung2.csv")
samsung2 <- samsung2$Close
samsung2 <- rev(samsung2)
sam_yld <- log(samsung2[1:528]/samsung2[2:529])
sam_yld <- rev(sam_yld)
sam_train <- sam_yld[1:344]
sam_test <- sam_yld[345:528]

msciem <- read.csv("msciem.csv", sep="")
msciem <- gsub(pattern=",", replacement="", x=msciem$Price)
msciem <- as.character(msciem)
msciem <- as.numeric(msciem)
msciem_yld <- log(msciem[1:528]/msciem[2:529])
msciem_yld <- rev(msciem_yld)
msciem_train <- msciem_yld[1:344]
msciem_test <- msciem_yld[345:528]
df_em2 <- data.frame(sam_yld, msciem_yld)
em_fit2 <- lm(sam_yld~msciem_yld, data=df_em2)
summary(em_fit2)

apple2<- read.csv("apple2.csv")
apple2 <- apple2$Close
apple2 <- rev(apple2)
apple_yld <- log(apple2[1:528]/apple2[2:529])
apple_yld <- rev(apple_yld)
apple_train <- apple_yld[1:344]
apple_test <- apple_yld[345:528]
df_apple2 <- data.frame(sam_yld, apple_yld)
apple_fit2 <- lm(sam_yld~apple_yld, data=df_apple2)
summary(apple_fit2)

sox2<- read.csv("sox2.csv")
sox2 <- sox2$Close
sox2 <- rev(sox2)
sox_yld <- log(sox2[1:528]/sox2[2:529])
sox_yld <- rev(sox_yld)
sox_train <- sox_yld[1:344]
sox_test <- sox_yld[345:528]
df_sox2 <- data.frame(sam_yld, sox_yld)
sox_fit2 <- lm(sam_yld~sox_yld, data=df_sox2)
summary(sox_fit2)

tencent<- read.csv("tencent.csv")
tencent <- tencent$Close
tencent <- rev(tencent)
tenc_yld <- log(tencent[1:528]/tencent[2:529])
tenc_yld <- rev(tenc_yld)
tenc_train <- tenc_yld[1:344]
tenc_test <- tenc_yld[345:528]
df_tenc2 <- data.frame(sam_yld, tenc_yld)
tenc_fit2 <- lm(sam_yld~tenc_yld, data=df_tenc2)
summary(tenc_fit2)

dax2<- read.csv("dax2.csv")
dax2 <- dax2$Close
dax2 <- rev(dax2)
dax_yld <- log(dax2[1:528]/dax2[2:529])
dax_yld <- rev(dax_yld)
dax_train <- dax_yld[1:344]
dax_test <- dax_yld[345:528]
df_dax2 <- data.frame(sam_yld, dax_yld)
dax_fit2 <- lm(sam_yld~dax_yld, data=df_dax2)
summary(dax_fit2)

df_total <- data.frame(sam_train, msciem_train,  apple_train, sox_train, tenc_train, dax_train)
tot_fit <- lm(sam_train~., data=df_total)
summary(tot_fit2)

df_total2 <- data.frame(sam_yld, msciem_yld, dax_yld, apple_yld, sox_yld, tenc_yld)
tot_fit2 <- lm(sam_yld~., data=df_total2) 
summary(tot_fit2)

## Regressione logistica

jpykrw <- read.csv("krwjpy.csv")
jpykrw <- jpykrw$Price
jpykrw_yld <- log(jpykrw[1:528]/jpykrw[2:529])
jpykrw_yld <- rev(jpykrw_yld)
samsungfactor <- ifelse(sam_yld < 0, 0,1)
samfactor <- as.factor(samsungfactor)
tencfactor <- factor(ifelse(tenc_yld < 0, 0,1))
applefactor <- factor(ifelse(apple_yld < 0, 0,1))
df_total3 <- data.frame(samfactor, msciem_yld, dax_yld, nasd_yld, jpykrw_yld)
df_total4 <- data.frame(tencfactor, msciem_yld, dax_yld, jpykrw_yld, nasd_yld)
df_total5 <- data.frame(applefactor, msciem_yld, dax_yld, jpykrw_yld, nasd_yld)
splitting<- createDataPartition(df_total3$samfactor, p=0.75, list=F)
splitting2<- createDataPartition(df_total4$tencfactor, p=0.75, list=F)
splitting3 <- createDataPartition(df_total5$applefactor, p=0.75, list=F)
train2 <- df_total3[splitting,]
test2 <- df_total3[-splitting,]
train3 <- df_total4[splitting2,]
test3 <- df_total4[-splitting2,]
train4 <- df_total5[splitting3,]
test4 <- df_total5[-splitting3,]
length(df_total3$samfactor[df_total3$samfactor == 0])
length(df_total4$tencfactor[df_total4$tencfactor == 0])
length(df_total5$applefactor[df_total5$applefactor == 0])
length(train2$samfactor)
length(train2$samfactor[train2$samfactor == 0])
length(train3$tencfactor[train3$tencfactor==0])
length(train3$tencfactor)
length(train4$applefactor[train4$applefactor==0])
length(train4$applefactor)
length(test2$samfactor[test2$samfactor == 0])
length(test2$samfactor)
length(test3$tencfactor[test3$tencfactor == 0])
length(test3$tencfactor)
length(test4$applefactor[test4$applefactor == 0])
length(test4$applefactor)
samsung_glm <- train(form=samfactor~., data=train2, trControl=trainControl(method="cv", number=5),
                     method="glm", family="binomial" )


tencent_glm <- train(form=tencfactor~., data=train3, trControl=trainControl(method="cv", number=5),
                     method="glm", family="binomial" )


apple_glm <- train(form=applefactor~., data=train4, trControl=trainControl(method="cv", number=5),
                   method="glm", family="binomial" )

summary(samsung_glm)
summary(tencent_glm)
summary(apple_glm)

confusionMatrix(samsung_glm, reference=train2$samfactor)
confusionMatrix(tencent_glm, reference=train3$tencfactor)
confusionMatrix(apple_glm, reference=train4$applefactor)

glm_testing <- predict(samsung_glm, newdata=test2)
glm_testing2 <- predict(tencent_glm, newdata=test3)
glm_testing3 <- predict(apple_glm, newdata=test4)

confusionMatrix(glm_testing, reference=test2$samfactor)
confusionMatrix(glm_testing2, reference=test3$tencfactor)
confusionMatrix(glm_testing3, reference=test4$applefactor)
