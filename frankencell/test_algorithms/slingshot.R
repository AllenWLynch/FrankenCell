#!/usr/local/bin/Rscript

args = commandArgs(trailingOnly=TRUE)

library(slingshot)
library(Matrix)
library(matrixStats)
library(jsonlite)

data <- fromJSON(args[1])

sds <- slingshot(data$coordinates, clusterLabels = data$clusters, 
                 start.clus=data$start_cluster, stretch = 2, 
                 end.clus = data$end_clusters)

write.csv(slingPseudotime(sds), 
    file = args[2])