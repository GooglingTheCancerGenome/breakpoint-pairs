require(StructuralVariantAnnotation)
require(TVTB)
require(S4Vectors)
require(rtracklayer)
require(UpSetR)

negative_set <- c("NoBP_NoBP",
                  "CR_CR", 
                  "CR_TrueSV", 
                  "TrueSV_TrueSV")

callers <-   c('gridss')

file_paths <- c()
names_vec <- c()
names_cnn <- c()
for (neg in negative_set){
  for (cal in callers){
    for (i in 1:10){
      file_paths <- c(file_paths, paste("/home/cog/smehrem/breakpoint-pairs/ConsensusCNN_VCF/",neg,"/FULLCalledSV_",cal, "_", i, ".vcf", sep=""))
      names_vec <- c(names_vec, paste(cal,"_",neg,"_",i, sep=""))
      names_cnn <- c(names_cnn, paste(cal,"_",neg,"_",i, sep=""))
    }
  }
}

file_paths <- c(file_paths, "/home/cog/smehrem/MinorResearchInternship/VCF/gridss.sym.vcf")
names_vec <- c(names_vec, "gridss_OG")

names(file_paths) <- names_vec

infoR <- VcfInfoRules(exprs = list(pr = expression(PR > 0.0)),active = c(TRUE))
infoR

gr <- list()
called_SV <- c()
for (sv_caller in names(file_paths))
{
  sv_callset_vcf <-
    VariantAnnotation::readVcf(as.vector(file_paths[sv_caller]))
  pr_support <- info(sv_callset_vcf)$PR
  #print(pr_support)
  if (sv_caller != "gridss_OG"){
  sv_callset_vcf_filtered <- S4Vectors::subsetByFilter(sv_callset_vcf, infoR)
  bpgr <- breakpointRanges(sv_callset_vcf_filtered) 
  begr <- breakendRanges(sv_callset_vcf_filtered)
  gr_temp <- sort(c(bpgr, begr))
  gr_temp <- gr_temp[gr_temp$svLen>=50]
  #print(length(gr_temp))
  gr[[sv_caller]] <- gr_temp

  called_SV <- c(called_SV, gr[[sv_caller]]$sourceId)
  }else{
    sv_callset_vcf <-
      VariantAnnotation::readVcf(as.vector(file_paths[sv_caller]))
 
    bpgr <- breakpointRanges(sv_callset_vcf) 
    begr <- breakendRanges(sv_callset_vcf)
    gr_temp <- sort(c(bpgr, begr))
    print(gr_temp[gr_temp$seqnames])
    print(length(gr_temp))
    gr_temp <- sort(c(bpgr, begr))
    gr_temp <- gr_temp[gr_temp$svLen>=50]
    print(length(gr_temp$svLen))
    gr[[sv_caller]] <- gr_temp
    called_SV <- c(called_SV, gr[[sv_caller]]$sourceId)
  }

}
print(length(called_SV))
called_SV <- unique(called_SV)
print(length(called_SV))
overlap_matrix <- data.frame(matrix(0L,ncol = length(names_vec)+2, nrow = length(called_SV)))
colnames(overlap_matrix) <- c(names_vec,"mills", "sv_classify")
rownames(overlap_matrix) <- called_SV
for (sv_caller in names_vec){
  sv_names <- gr[[sv_caller]]$sourceId
  for (name in sv_names){
    overlap_matrix[name, sv_caller] <- 1
  }
}



p <-rtracklayer::import("/home/cog/smehrem/MinorResearchInternship/VCF/lumpy-Mills2012-call-set.bedpe")

gr$mills <- pairs2breakpointgr(p)
seqlevelsStyle(gr$mills) <- "NCBI"
gr$mills <- sort(gr$mills)

p <-
  rtracklayer::import("/home/cog/smehrem/MinorResearchInternship/VCF/Personalis_1000_Genomes_deduplicated_deletions.bedpe")
gr$sv_classify <-
  pairs2breakpointgr(p, placeholderName = "sv_classify")
gr$sv_classify <- sort(gr$sv_classify)

for (n in names(gr))
{
  gr[[n]] <- gr[[n]][seqnames(gr[[n]]) %in% c("1", "2", "3")]
}

df <- data.frame()
for (gt_dataset in c('mills', 'sv_classify'))
{
  #gt_dataset <- 'sv_classify'
  ground_truth <- gr[[gt_dataset]]
  
  for (sv_caller in names(gr)[names(gr) %in% names_vec])
  {
    #sv_caller <- "gridss_TrueSV_TrueSV_1"
    sv_callset <- gr[[sv_caller]]
    overlaps <-
      countBreakpointOverlaps(ground_truth,
                              sv_callset,
                              maxgap = 200,
                              ignore.strand = TRUE,
                              sizemargin=0.25,
                              restrictMarginToSizeMultiple=0.5)


    names_overlap <- ground_truth[which(overlaps > 0)]$sourceId
    
    for (name in names_overlap){
      overlap_matrix[name,gt_dataset] <- 1
    }

    precision <-
      round(length(overlaps[overlaps > 0]) / length(sv_callset), digits = 2)
    recall <-
      round(length(overlaps[overlaps > 0]) / length(ground_truth), digits = 2)
    f1_score <-
      round(2 * (precision * recall) / (precision + recall), digits = 2)
    df <-
      rbind(
        df,
        data.frame(
          sv_caller = sv_caller,
          filtered = "PASS",
          ground_truth = gt_dataset,
          precision = precision,
          recall = recall,
          f1_score = f1_score,
          TP = length(overlaps[overlaps > 0]),
          FP = length(sv_callset) - length(overlaps[overlaps > 0]),
          FN = length(ground_truth) - length(overlaps[overlaps > 0])
        )
      )
  }
}


df <- df[order(df$f1_score, decreasing = TRUE),]
#write.table(df, file = paste("OverlapGridss_GridssCNN_GrdTrth_chr_1_to_3_PRbigger_0_99.csv", sep = "\t"))
df

