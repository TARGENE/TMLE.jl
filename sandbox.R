library(data.table)
library(tmle3)
library(sl3)

data = read.csv("/Users/olivierlabayle/Dev/TARGENE/TMLE.jl/test/data/perinatal.csv")

node_list <- list(
  W = c(
    "apgar1", "apgar5", "gagebrth", "mage", "meducyrs", "sexn"
  ),
  A = "parity01",
  Y = "haz01"
)

glm = Lrnr_glm$new()
lrn_mean = Lrnr_mean$new()
sl <- Lrnr_sl$new(learners = Stack$new(glm, lrn_mean), metalearner = Lrnr_nnls$new())

learner_list <- list(A = glm, Y = glm)
# learner_list = list(A=sl, Y = sl)

ate_spec <- tmle_ATE(
  treatment_level = 1,
  control_level = 0
)

tmle_task <- ate_spec$make_tmle_task(data, node_list)
initial_likelihood <- ate_spec$make_initial_likelihood(
  tmle_task,
  learner_list
)

targeted_likelihood_cv <- Targeted_Likelihood$new(initial_likelihood)

targeted_likelihood_no_cv <-
  Targeted_Likelihood$new(initial_likelihood,
    updater = list(cvtmle = FALSE)
  )

tmle_params_cv <- ate_spec$make_params(tmle_task, targeted_likelihood_cv)
tmle_params_no_cv <- ate_spec$make_params(tmle_task, targeted_likelihood_no_cv)

tmle_no_cv <- fit_tmle3(
  tmle_task, targeted_likelihood_no_cv, tmle_params_no_cv,
  targeted_likelihood_no_cv$updater
)
tmle_no_cv

tmle_cv <- fit_tmle3(
  tmle_task, targeted_likelihood_cv, tmle_params_cv,
  targeted_likelihood_cv$updater
)
tmle_cv