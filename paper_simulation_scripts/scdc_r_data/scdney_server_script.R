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
py_cellTypes <- read.delim("scdc_cellTypes.txt", header = FALSE, as.is=TRUE)$V1
py_subject <- read.delim("scdc_subject.txt", header = FALSE, as.is=TRUE)$V1
py_condition <- read.delim("scdc_condition.txt", header = FALSE, as.is=TRUE)$V1
py_short_conditions <- read.delim("scdc_short_conditions.txt", header = FALSE, as.is=TRUE)$V1

py_res_scDC_noClust <- scDC_noClustering(py_cellTypes, py_subject, calCI = TRUE, 
                                        calCI_method = c("BCa"),
                                        nboot = 100)

py_res_GLM <- fitGLM(py_res_scDC_noClust, py_short_conditions, pairwise = FALSE)

sum <- summary(py_res_GLM$pool_res_random)

write.csv(sum, "scdc_summary.csv")
