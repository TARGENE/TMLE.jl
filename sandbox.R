library(data.table)
library(tmle3)
library(sl3)
data <- fread(
  paste0(
    "https://raw.githubusercontent.com/tlverse/tlverse-data/master/",
    "wash-benefits/washb_data.csv"
  ),
  stringsAsFactors = TRUE
)

node_list <- list(
  W = c(
    "month", "aged", "sex", "momage", "momedu",
    "momheight", "hfiacat", "Nlt18", "Ncomp", "watmin",
    "elec", "floor", "walls", "roof", "asset_wardrobe",
    "asset_table", "asset_chair", "asset_khat",
    "asset_chouki", "asset_tv", "asset_refrig",
    "asset_bike", "asset_moto", "asset_sewmach",
    "asset_mobile"
  ),
  A = "tr",
  Y = "whz"
)

processed <- process_missing(data, node_list)
data <- processed$data
node_list <- processed$node_list

ate_spec <- tmle_ATE(
  treatment_level = "Nutrition + WSH",
  control_level = "Control"
)
tmle_spec = ate_spec

# choose base learners
lrnr_mean <- make_learner(Lrnr_mean)
lrnr_rf <- make_learner(Lrnr_ranger)

# define metalearners appropriate to data types
ls_metalearner <- make_learner(Lrnr_nnls)
mn_metalearner <- make_learner(
  Lrnr_solnp, metalearner_linear_multinomial,
  loss_loglik_multinomial
)
sl_Y <- Lrnr_sl$new(
  learners = list(lrnr_mean, lrnr_rf),
  metalearner = ls_metalearner
)
sl_A <- Lrnr_sl$new(
  learners = list(lrnr_mean, lrnr_rf),
  metalearner = mn_metalearner
)
learner_list <- list(A = sl_A, Y = sl_Y)

tmle_task <- tmle_spec$make_tmle_task(data, node_list)

initial_likelihood <- tmle_spec$make_initial_likelihood(tmle_task, learner_list)

updater <- tmle_spec$make_updater()
targeted_likelihood <- tmle_spec$make_targeted_likelihood(initial_likelihood, updater)

tmle_params <- tmle_spec$make_params(tmle_task, targeted_likelihood)
updater$tmle_params <- tmle_params

fit <- fit_tmle3(tmle_task, targeted_likelihood, tmle_params, updater)

update_fold <- updater$update_fold
maxit <- 100

# seed current estimates
current_estimates <- lapply(updater$tmle_params, function(tmle_param) {
    tmle_param$estimates(tmle_task, update_fold)
})

likelihood = initial_likelihood
likelihood_values <- likelihood$cache$get_values(A_factor, tmle_task, fold_number)
likelihood_values <- A_factor$get_likelihood(tmle_task, fold_number)

update_node <- updater$update_nodes[[1]]
submodel_data <- updater$generate_submodel_data(
          likelihood, tmle_task,
          fold_number, update_node,
          drop_censored = TRUE
        )

for (steps in seq_len(maxit)) {
    updater$update_step(likelihood, tmle_task, update_fold)

    # update estimates based on updated likelihood
    private$.current_estimates <- lapply(self$tmle_params, function(tmle_param) {
        tmle_param$estimates(tmle_task, update_fold)
    })

    if (self$check_convergence(tmle_task, update_fold)) {
        break
    }

    if (self$use_best) {
        self$update_best(likelihood)
    }
}

if (self$use_best) {
    self$update_best(likelihood)
    likelihood$cache$set_best()
}

private$.steps <- self$updater$steps


estimates <- lapply(
    self$tmle_params,
    function(tmle_param) {
        tmle_param$estimates(self$tmle_task, self$updater$update_fold)
    }
)

private$.estimates <- estimates
private$.ED <- ED_from_estimates(estimates)

