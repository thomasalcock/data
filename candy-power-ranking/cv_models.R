
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
library(leaps)

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
  geom_bar(stat = "identity", alpha = 0.8) +
  theme_minimal() + labs(
    x = "Relevanz", 
    y = "Eigenschaft", 
    fill = "",
    title = "Modell: LASSO"
  )

# Lasso Predictions -------------------------------------------------------

get_model_perf <- function(wlf, df, model) {

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

perf_plot_lasso <- get_model_perf(final_lasso, model_df, "LASSO")

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

vi_plot <- function(final_model, df, model){
  final_model %>% 
    fit(data = df) %>% 
    pull_workflow_fit() %>% 
    vi() %>% ggplot(aes(x = Importance, y = fct_reorder(Variable, Importance))) +
    geom_bar(stat = "identity", fill = "blue", alpha = 0.7) +
    theme_minimal() + labs(
      x = "Relevanz", 
      y = "Eigenschaft", fill = "",
      title = paste("Modell:", model)
    )
}

vimp_ranger <- vi_plot(final_ranger, model_df, "Random Forest")
  
# Ranger Predictions ------------------------------------------------------

perf_plot_ranger <- get_model_perf(final_ranger, model_df, "Random Forest")


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

perf_plot_xbg <- get_model_perf(final_xgb, model_df, "XGBoost")

vi_xbg <- vi_plot(final_xgb, model_df, "XGBoost")

# Reduced XGBoost ---------------------------------------------------------

rev_xgb <- boost_tree(
  mode = "regression",
  mtry = 7,
  trees = 91,
  learn_rate = 0.037
) %>% 
  set_engine(engine = "xgboost", importance = "impurity") %>% 
  fit(winpercent ~ chocolate + pricepercent + sugarpercent + fruity + peanutyalmondy, data = model_df)

perf_plot(model_df, rev_xgb)

sim_xbg <- function(choc, price, sugar, fruit, pean, model){
  
  df <- crossing(
    chocolate = choc, pricepercent = price,
    fruity = fruit, sugarpercent = sugar,
    peanutyalmondy = pean
  )
  
  get_preds(df, model)
}

xgbsim <- sim_xbg(
  choc = c(TRUE,FALSE),
  price = seq(min(model_df$pricepercent), max(model_df$pricepercent), 0.01), 
  sugar = seq(min(model_df$sugarpercent), max(model_df$sugarpercent), 0.01), 
  fruit = c(TRUE,FALSE), 
  pean = c(TRUE, FALSE), 
  model = rev_xgb
)

best_pred <- xgbsim[which.max(xgbsim$.pred),]

model_df %>% 
  filter(chocolate, !fruity, peanutyalmondy) %>% 
  filter(abs(pricepercent - best_pred$pricepercent) < sim_threshold) %>% 
  pull(competitorname)

xgbsim %>%
  filter(abs(sugarpercent - 0.701) < 0.00001) %>% 
  filter(!fruity, peanutyalmondy) %>% 
  ggplot(aes(x = pricepercent, y = .pred, color = chocolate)) + 
  geom_line() + theme_minimal() + labs(y = "Predicted Winpercent", x = "Pricepercent")

# method: 
# 1. find good fitting model
# 2. simulate all possible scenarios
# 3. get scenario with maximum winpercent
# 4. find candy closest to that scenario for recommendation
# TODO implement for linear model

# Best Subset selection --------------------------------------------------

ex_selection <- regsubsets(
  x = winpercent ~ .,
  data = candy_df %>% select(-competitorname) %>% as.data.frame(),
  nvmax = NULL,
  intercept = TRUE
)

get_best_vars <- function(regsubset_results){
  summary_selection <- summary(ex_selection)
  selected_vars <- summary_selection$out[which.max(summary_selection[["adjr2"]]),]
  selected_vars <- names(selected_vars[which(selected_vars == "*")])
  gsub("TRUE", "", selected_vars)
}

selected_feats <- get_best_vars(ex_selection)


# LM with Best Subset selection ----------------------------------------

coef_plot <- function(model){
  
  tidy(model) %>%
    mutate(term = gsub("TRUE", "", term),
           term = fct_reorder(term, abs(statistic))) %>% 
    filter(term != "(Intercept)") %>% 
    ggplot(aes(x = estimate, y = term)) + 
    geom_point() + 
    geom_errorbar(aes(xmin = estimate - 1.96*std.error,
                      xmax = estimate + 1.96*std.error)) +
    theme_minimal() +
    geom_vline(xintercept = 0, linetype = 2) +
    labs(x = "Point Estimate", y = "Eigenschaft")
  
}

get_preds <- function(df, model){
  df %>% 
    bind_cols(predict(model, df))
}

perf_plot <- function(df, model){
  
  pred_data <- get_preds(df, model)
  
  model_mae <- mae(pred_data, winpercent, .pred)
  model_rmse <- rmse(pred_data, winpercent, .pred)
  model_rsq <- rsq_trad(pred_data, winpercent, .pred)
  
  preds_lm %>% ggplot(aes(x = winpercent, y = .pred)) +
    geom_point(alpha = 0.7, size = 3) + geom_smooth(se = FALSE) +
    theme_minimal() + labs(x = "Observed", y = "Predicted",
                           title = paste0(
                             "Modell: LM \n",
                             "RMSE: ", round(model_rmse$.estimate, 1), "\n",
                             "MAE: ", round(model_mae$.estimate, 1), "\n",
                             "R^2: ", round(model_rsq$.estimate, 1)
                           ))
}

summary_plots <- function(model, df){
  grid.arrange(
    perf_plot(df, model), 
    coef_plot(model),
    ncol = 2
  )
}

lin_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  fit(winpercent ~ ., data = candy_df %>% 
        select(-competitorname, winpercent, all_of(selected_feats)))

summary_plots(lin_model, candy_df)

# refine lm
new_feats <- tidy(lin_model) %>% 
  filter(term != "(Intercept)") %>% 
  filter(p.value < 0.08) %>% 
  pull(term)

lin_model_rev <- linear_reg() %>% 
  set_engine("lm") %>% 
  fit(as.formula(paste0("winpercent~",paste0(new_feats,collapse="+"))), data = candy_df %>% 
        select(-competitorname, winpercent, all_of(new_feats)))

summary_plots(lin_model_rev, candy_df)

# VIMP Comparison ---------------------------------------------------------

grid.arrange(vimp_lasso, vimp_ranger, vi_xbg, nrow = 2)

# Test model for sensibilty -----------------------------------------------

preddf <- get_preds(candy_df, lin_model)

simulate_win <- function(choc, pean, fruit, hard, sugarp){
  
  sim_data <- crossing(
    chocolate = choc,
    peanutyalmondy = pean,
    fruity = fruit,
    hard = hard,
    sugarpercent = sugarp
  )
  
  sim_data %>% 
    predict(lin_model_rev, .) %>% 
    bind_cols(sim_data) %>% .$.pred
}



# Pick most suitable candy ------------------------------------------------

# 1. step, candy must have most important attr: choc, peanut, sugarpercent

preddf %>% 
  filter(chocolate & peanutyalmondy & !hard) %>%
  filter(pricepercent <= quantile(pricepercent, probs = 0.25)) %>% 
  filter(bar & crispedricewafer) %>% pull(competitorname)


