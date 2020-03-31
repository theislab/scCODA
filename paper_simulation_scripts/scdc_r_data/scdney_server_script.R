# SCDC (U. of Sydney)  Simulation for model comparison

library(scdney)
library(tidyverse)

getCurrentFileLocation <-  function()
{
  this_file <- commandArgs() %>% 
    tibble::enframe(name = NULL) %>%
    tidyr::separate(col=value, into=c("key", "value"), sep="=", fill='right') %>%
    dplyr::filter(key == "--file") %>%
    dplyr::pull(value)
  if (length(this_file)==0)
  {
    this_file <- rstudioapi::getSourceEditorContext()$path
  }
  return(dirname(this_file))
}
setwd(getCurrentFileLocation())

# Load data
# data_path = "/home/icb/johannes.ostner/compositional_diff/compositionalDiff-johannes_tests_2/paper_simulation_scripts/scdc_r_data/"
data_path = ""

py_cellTypes <- read.delim(paste(data_path, "scdc_cellTypes.txt", sep=""), header = FALSE, as.is=TRUE)$V1
py_subject <- read.delim(paste(data_path, "scdc_subject.txt", sep=""), header = FALSE, as.is=TRUE)$V1
py_condition <- read.delim(paste(data_path, "scdc_condition.txt", sep=""), header = FALSE, as.is=TRUE)$V1
py_short_conditions <- read.delim(paste(data_path, "scdc_short_conditions.txt", sep=""), header = FALSE, as.is=TRUE)$V1

py_res_scDC_noClust <- scDC_noClustering(py_cellTypes, py_subject, calCI = TRUE, 
                                        calCI_method = c("BCa"),
                                        nboot = 100)

py_res_GLM <- fitGLM(py_res_scDC_noClust, py_short_conditions, pairwise = FALSE)

sum <- summary(py_res_GLM$pool_res_random)

print(sum)

write.csv(sum, paste(data_path, "scdc_summary.csv", sep=""))
