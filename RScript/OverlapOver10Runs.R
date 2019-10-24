require(StructuralVariantAnnotation)
require(TVTB)
require(S4Vectors)
require(rtracklayer)
library(ggplot2)

negative_set <- c("NoBP_NoBP",
                  "CR_CR", 
                  "CR_TrueSV", 
                  "TrueSV_TrueSV")

callers <-   c('delly',
               'gridss',
               'lumpy',
                'manta')

file_paths <- c()
names_vec <- c()

for (neg in negative_set){
  for (cal in callers){
    for (i in 1:10){
      if (cal == "gridss"){
        file_paths <- c(file_paths, paste("/home/cog/smehrem/breakpoint-pairs/ConsensusCNN_VCF/",neg,"/NEWCalledSV_",cal, "_", i, ".vcf", sep=""))
        names_vec <- c(names_vec, paste(cal,"_",neg,"_",i, sep=""))
      }else{
      file_paths <- c(file_paths, paste("/home/cog/smehrem/breakpoint-pairs/ConsensusCNN_VCF/",neg,"/CalledSV_",cal, "_", i, ".vcf", sep=""))
      names_vec <- c(names_vec, paste(cal,"_",neg,"_",i, sep=""))
    }
  }
  }
}

names(file_paths) <- names_vec

infoR <- VcfInfoRules(exprs = list(pr = expression(PR > 0)),active = c(TRUE))
infoR

gr <- list()
pr_df<- list()

for (sv_caller in names(file_paths))
{
  sv_callset_vcf <-
    VariantAnnotation::readVcf(as.vector(file_paths[sv_caller]))
  sv_callset_vcf_filtered <- S4Vectors::subsetByFilter(sv_callset_vcf, infoR)
  bpgr <- breakpointRanges(sv_callset_vcf_filtered) 
  begr <- breakendRanges(sv_callset_vcf_filtered)
  gr_temp <- sort(c(bpgr, begr))
  gr_temp <- gr_temp[gr_temp$svLen>=50]
  gr[[sv_caller]] <- gr_temp
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
  ground_truth <- gr[[gt_dataset]]

    for (sv_caller in names(gr)[names(gr) %in% names_vec])
    {

      sv_callset <- gr[[sv_caller]]
      overlaps <-
        countBreakpointOverlaps(sv_callset,
                                ground_truth,
                                maxgap = 200,
                                ignore.strand = TRUE)
      
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
df
#write.table(df, file = "OverlapAllCallerswithMillsSVClassify_chr_1_to_3_PRbigger_0_90.csv", sep = "\t")

