import matrix from '../images/quarterly_data_correlation_matrix.png';
import quarter_model_one from '../images/quarterly_model_one_summary.png';
import model_one from '../images/original_data_vs_prediction.png';
import quarter_model_two from '../images/model_two_summary.png';
import model_two from '../images/original_data_vs_prediction_model_two.png';
import quarter_model_three from '../images/model_three_summary.png';
import model_three from '../images/original_data_vs_prediction_model_three.png';
import elastic_net from '../images/elastic_net_regression.png';

export default function Models() {
    return (
    <div>
        {/* <h1>Models</h1> */}

        {/* DATA WRANGLING SECTION */}
        <h1>Data Wrangling</h1>
        <div class="container-model">
            <div class="img-col">
                <img src={matrix} alt="alt-logo"/>
                <p className='models-p'>Figure 1.1 Quarterly Correlation Matrix - This Figure shows the 
                    correlation and how strong that correlation is between variables 
                    used in the quarterly model</p>
            </div>

            <div class="text-col">
                <h2>Description of Model</h2>
                <p className="models-p">
                Like the yearly sections above, we will again wrangle the data into a 
                format that can be used for modeling. This would include joining the 
                different data into one dataset. This step was much easier than the yearly 
                data since we got each variable from FRED, which would provide us a csv 
                file that would have only the date in one column and the variable such as 
                GDP, CPI in the other column. The only real cleaning that needed to be 
                done was changing the column names that were giving joining the tables 
                some issues. Unlike the yearly data which had 20 rows, after joining all 
                the tables by date, we had over one hundred rows of data points, which is 
                a significant increase in data. Another data wrangling step that was done 
                for quarterly data, but not yearly data, was splitting the data into 
                training and testing datasets. This split would be used to better evaluate 
                the models by testing the model on data that it has never seen before. 
                The test/train split was seventy percent training and thirty percent 
                testing. Another step that was done was making sure that the testing data 
                had the most recent quarterly data, this would include Q1, Q2 of 2022. 
                This was done so that we can see how well our model does in a recent time 
                frame, making it more relevant to today. The last data wrangling step that 
                was done was normalizing the data, however, this did not have much of an 
                affect on the models so we won't be discussing it further and we didn't 
                end up sticking with keeping our data normalized.
                </p>
            </div>
        </div>

        {/* LINEAR REGRESSION SECTION */}
        {/* SECTION ONE */}
        <h1>Linear Regression</h1>
        <div className="container-model">
            <div class="img-col">
                <img src={quarter_model_one} alt="alt-logo"/>
                <p className='models-p'>Figure 1.2 Quarterly Model 1 Summary - This figure 
                    shows the summary of quarterly model 1
                </p>

                <img src={model_one} alt="model-one"/>
                <p className='models-p'>Figure 1.3 Model 1 Original Vs Prediction - This figure shows a graph 
                    with the original data and the predicted values of quarterly model 1
                </p>
            </div>

            <div class="text-col">
                {/* <h2>Figure 1.2 Quarterly Model 1 Summary - This figure shows the summary 
                    of quarterly model 1</h2> */}
                <h2>Description of Figure 1.2 and Figure 1.3</h2>

                <p className="models-p">
                Once the quarterly data was collected and compiled into a usable dataset, 
                modeling can begin. The first model that we will look at uses all the 
                variables that are in our dataset. The formula can be seen at the top of 
                Figure 7.1.6 above. This summary also gives us coefficients which tell us 
                how to create the line that the model creates along with the intercept. 
                For example we can say that for every increase in GDP along the x-axis, 
                cpi will decrease by -1.853e-01 and so on for the other variables. The 
                next statistic to look at is the adjusted R-squared value which can show 
                us how well our model can explain the variance in the dependent variable 
                (GDP) by the movement of the independent variables 
                (CPI, unemployment claims, etc). The adjusted R-squared value tries to 
                adjust to the multiple variables that we use, since the more variables 
                we have we expect to be able to account for more variance. Both the 
                adjusted and normal R-squared values are around 0.8 or 80%, which is very 
                good. The model seems to be performing very well along with the fact that 
                we have a low residual standard error (0.4827). A low residual standard is 
                what you want for a model since the smaller value means that the model 
                does better at predictions than a model with a higher value. <br></br>

                <br></br>
                Comparing this first quarterly model to the yearly model, we can see that 
                the adjusted R-squared value is much smaller in this model than that of 
                the yearly one. However, in the conclusion of the yearly model, we said 
                that we believed that the model was overfitted to the training data which 
                gave us a high R-squared value, but when we looked at the residual 
                standard error (231.3, see Figure 7.1.3) it is much higher than this 
                model (0.4827). This shows that this model is not overfitted, unlike the 
                yearly model. We can see below in Figure 7.1.7 when we test our model on 
                data that it has never seen before it does seem to get the general trend 
                of the data, although it does seem to not account for all the variance, 
                which is why for this model the adjusted R-squared value is in the 80% 
                and not higher. See below in the our models vs others section for seeing 
                what this and the other models below predicted percent gdp growth for 
                quarter 1, 2 of 2022.

                </p>
            </div>
        </div>

        {/* SECTION TWO */}
        {/* <h2>Figure 1.4 Quarterly Model 2 Summary - This figure shows the summary of 
            quarterly model 2</h2> */}
        <div class="container-model">
            <div class="img-col">
                <img src={quarter_model_two} alt="alt-logo"/>
                <p className='models-p'>Figure 1.4 Quarterly Model 2 Summary - This 
                figure shows the summary of quarterly model 2</p>

                <img src={model_two} alt="alt-logo"/>
                <p className='models-p'>Figure 1.5 Model 2 Original data Vs Prediction - 
                This figure shows a graph with the original data and the predicted values 
                of quarterly model 2</p>
            </div>

            <div class="text-col">
                <h2>Description of Figures 1.4 and Figure 1.5</h2>
                <p className="models-p">
                This second model we wanted to see what would happen if we just used the 
                highly correlated variable to predict GDP. The formula can be seen at the 
                top of Figure 7.1.8 above. This summary also gives us coefficients which 
                tell us how to create the line that the model creates along with the 
                intercept. For example we can say that for every increase in GDP along the 
                x-axis, cpi will decrease by -0.39374 and so on for the other variables. 
                The next statistic to look at is the adjusted R-squared value which is 
                horrible. It means that we cannot account for variance in GDP given our 
                independent variables. This is not something I would think would happen, 
                since these variables are already highly correlated with the dependent 
                variable. Even though the model has a low R-squared value the model has a 
                low residual standard error (1.015). If we recall, a low residual standard 
                is what you want for a model since the smaller value means that the model 
                does better at predictions than a model with a higher value. Given that 
                the adjusted R-squared value is so low, we would not expect the model to 
                predict very well, however, when we looked at the predicted values this 
                model gave us, it was surprisingly accurate at predicting the quarter 1 
                and 2 of 2022. <br></br><br></br>
                
                When we compare this model to the yearly and quarterly model 1, this model 
                does not look impressive, in fact it looks pretty horrible. It is hard to 
                say if this model is better than the yearly model, since that model is 
                overfitted and has a very high residual standard error, but this model's 
                R-squared value is extremely low. The details of this model honestly is 
                not the interesting part, it's the fact that given the model is so bad, 
                how did it predict so well, which will be discussed below. Looking at 
                Figure 7.1.9 below we can see that the model does seem to get some of the 
                variance but it hardly performs and it is very noticeable to the eye.

                </p>
            </div>
        </div>

        <div class="container-model">
            <div class="img-col">
                <img src={quarter_model_three} alt="alt-logo"/>
                <p className='models-p'>Figure 1.6 Quarterly Model 3 Summary - This figure shows the 
                    summary of quarterly model 3
                </p>

                <img src={model_three} alt="alt-logo"/>
                <p className='models-p'>Figure 1.7 Model 3 Original data Vs Prediction - 
                This figure shows a graph with the original data and the predicted values 
                of quarterly model 3</p>
            </div>

            <div class="text-col">
                <h2>Description of Figures 1.6 and Figure 1.7
                </h2>
                <p className="models-p">
                The third quarterly model that we are showing is one where we wanted to 
                use the least amount of variables, while still performing well, that being
                having a high R-squared and a low residual standard error. The variables 
                that were removed were the highly correlated variables 
                (darker boxes in Figure 7.1.1), or the variables that we used in quarterly 
                model 2 (Figure 7.1.8). If we compare this model to quarterly model 1 
                (Figure 7.1.6) we see that they are very similar. The residual standard 
                error is very low and a very similar value to that of this model. This 
                means that like the first model, this one should predict value well. The 
                R-squared value is also almost the same, meaning that we account for 
                almost the same variance of GDP given less variables used. This is 
                impressive since we went from 8 variables to 5 variables and are still 
                getting models roughly the same. <br></br><br></br>

                When we look at what the model predicts in Figure 7.1.11, we can see 
                visually that the model does seem to get the general trends like what we 
                see in quarterly model 1. This model may even look better to the eye. We 
                will compare the quarter 1, 2 of 2022 predictions below.
                </p>
            </div>
        </div>

        {/* ELASTIC-NET REGRESSION */}
        <h1>Elastic-Net Regression</h1>
        <div className="container-model">
            <div class="img-col">
                <img src={elastic_net} alt="alt-logo"/>
                <p className='models-p'>Figure 1.8 Elastic-Net Original data Vs 
                    Prediction - This figure shows a graph with the original data and the 
                    predicted values of quarterly model 1 of elastic net
                </p>
            </div>

            <div class="text-col">
                <h2>Description of Figure 1.8</h2>
                <p className="models-p">
                Elastic net was chosen as the second algorithm to try and compare to the 
                multivariable linear regression models. This was done using Python instead 
                of R, which is why we don't have a nice summary to look at. Although the 
                most important parts are that the Elastic-new model had a very high 
                R-squared value (0.9427 or 94%), indicating that the model can explain 
                the variance in the dependent variable given the independent variables. 
                The RMSE (root mean square error is rather large of 768, indicating that 
                the model seems to be relatively off when predicting values, and we will 
                take a look at the predicted values for quarter 2 of 2022 since quarter 1 
                could not be gotten. Like the quarterly models above, the data for 
                modeling was split into training and testing, that way when we test, the 
                model has never seen these values before. <br></br><br></br>

	            When looking at the graph below (Figure 7.1.13) we can see that the 
                elastic-net model did very well in showing the variance in the data. 
                When we look at the graph, when the GDP is low the model predicts low and 
                when the GDP is high the model accurately predicts a high GDP. This can be 
                seen very well visually through the graph below. However, how well will 
                this model do compared to our other models and to what others might predict.
                </p>
            </div>
        </div>

    

    </div>
    )
}