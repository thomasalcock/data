
# Libraries ---------------------------------------------------------------

library(glmnet)
library(tidymodels)
library(dplyr)
library(tidyr)
library(tibble)
library(parsnip)
library(recipes)
library(tune)
library(rsample)
library(tune)
library(dials)
library(vip)
library(forcats)
library(yardstick)
library(doParallel)

# Load data ---------------------------------------------------------------

set.seed(1)

candy_df <- read.csv(
  file = "candy-data.csv",
  header = TRUE
) %>% as_tibble()

model_df <- candy_df %>% 
  mutate_if(is.integer, as.logical)

folds <- vfold_cv(model_df, v = 20)


# Define recipe -----------------------------------------------------------

model_recipe <- recipe(winpercent ~ ., data = model_df) %>% 
  update_role(competitorname, new_role = "ID") %>% 
  step_normalize(sugarpercent, pricepercent)

wf <- workflow() %>% 
  add_recipe(model_recipe)

# Lasso CV -------------------------------------------------------------------

lasso <- linear_reg(mode = "regression", penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") 

# Model training over CV folds
res_lasso <- tune_grid(
  wf %>% add_model(lasso), 
  resamples = folds, 
  grid = 30,
  metrics = metric_set(mae, rmse)
)

tune_df <- res_lasso %>% 
  collect_metrics()


# Lasso tune plot ---------------------------------------------------------

tune_plot <- function(df, err_metric = "mae", param, x_log_10 = TRUE){
  p <- df %>% 
    filter(.metric == err_metric) %>% 
    ggplot(aes(x = !!sym(param), y = mean, color = .metric)) +
    geom_errorbar(
      aes(ymin = mean - std_err,
          ymax = mean + std_err),
      alpha = 0.5
    ) +
    geom_line(aes(y = mean)) +
    theme_minimal()
  
  if(x_log_10){
    p + scale_x_log10()
  }else(
    p
  )
}

grid.arrange(
  tune_plot(tune_df, param = "penalty"),
  tune_plot(tune_df, param = "mixture")
)


# Select best lasso by RMSE -----------------------------------------------

best_lasso <- res_lasso %>% 
  select_best("mae")

final_lasso <- finalize_workflow(
  wf %>% add_model(lasso),
  best_lasso
)

# VI Lasso ----------------------------------------------------------------

vimp_lasso <- final_lasso %>% 
  fit(data = model_df) %>% 
  pull_workflow_fit() %>% 
  vi(lambda = best_lasso$penalty) %>% 
  mutate(Sign = if_else(Sign == "NEG", "Negativer Effekt", "Positiver Effekt")) %>% 
  ggplot(aes(x = Importance, y = fct_reorder(Variable, Importance), fill = Sign)) +
  geom_bar(stat = "identity") +
  theme_minimal() + labs(
    x = "Relevanz", y = "Eigenschaft", fill = "",
    title = "Relevanz von Eigenschaften für Beliebtheit"
  )

vimp_lasso

# Lasso Predictions -------------------------------------------------------

get_model_perf <- function(wlf, df, model) {
  pred_data <- wlf %>% 
    fit(data = df) %>% 
    predict(df) %>% 
    bind_cols(df)
  
  model_rmse <- rmse(pred_data, winpercent, .pred)
  model_mae <- mae(pred_data, winpercent, .pred)
  
  plot_preds <- pred_data %>% 
    ggplot(aes(x = winpercent, y = .pred)) +
    geom_point(size = 5, alpha = 0.5, color = "blue") + 
    theme_minimal() +
    labs(x="Observed", y="Predicted", 
         title=paste0(
           "Modell: ", model, "\n",
           "RMSE: ", round(model_rmse$.estimate, 1), "\n",
           "MAE: ", round(model_mae$.estimate, 1))
         ) +
    geom_smooth(se = FALSE)
  
  plot_preds
}

get_model_perf(final_lasso, model_df, "LASSO")

# Ranger CV ------------------------------------------------------------------

ranger_model <- rand_forest(
  mode = "regression",
  mtry = tune(), 
  trees = tune(), 
  min_n = tune()
) %>% set_engine(
  engine = "ranger",
  importance = "impurity"
)

registerDoParallel()

res_ranger <- tune_grid(
  wf %>% add_model(ranger_model),
  resamples = folds,
  grid = 20,
  metrics = metric_set(mae, rmse)
)


# Ranger tune plot --------------------------------------------------------

tune_ranger <- res_ranger %>% 
  collect_metrics()

# Select Best Ranger ------------------------------------------------------

best_ranger <- res_ranger %>% 
  select_best("mae")

final_ranger <- finalize_workflow(
  wf %>% add_model(ranger_model),
  best_ranger
)


# Ranger VI ---------------------------------------------------------------

vimp_ranger <- final_ranger %>% 
  fit(data = model_df) %>% 
  pull_workflow_fit() %>% 
  vi() %>% 
  ggplot(aes(x = Importance, y = fct_reorder(Variable, Importance))) +
  geom_bar(stat = "identity") +
  theme_minimal() + labs(
    x = "Relevanz", y = "Eigenschaft", fill = "",
    title = "Relevanz von Eigenschaften für Beliebtheit"
  )

grid.arrange(vimp_lasso, vimp_ranger)


# Ranger Predictions ------------------------------------------------------

get_model_perf(final_ranger, model_df, "Random Forest")


# Xgboost CV -----------------------------------------------------------------

xgb_model <- boost_tree(mode = "regression", mtry = tune(), trees = tune(), learn_rate = tune()) %>% 
  set_engine(engine = "xgboost", importance = "impurity")

xgb_wf <- workflow() %>% 
  add_recipe(model_recipe) %>% 
  add_model(xgb_model)

registerDoParallel()

xgb_res <- tune_grid(
  xgb_wf,
  resamples = folds,
  grid = 10,
  metrics = metric_set(rmse, mae)
)

xgb_tune_df <- collect_metrics(xgb_res)

best_xgb <- select_best(xgb_res, metric = "mae")

final_xgb <- finalize_workflow(
  wf %>% add_model(xgb_model),
  best_xgb
)

get_model_perf(final_xgb, model_df, "XGBoost")
