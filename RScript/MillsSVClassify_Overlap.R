require(StructuralVariantAnnotation)
require(TVTB)
require(S4Vectors)
require(rtracklayer)
gr <- list()
names_vec <- c("mills", "sv_classify")
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
for (gt_dataset in names_vec)
{
  ground_truth <- gr[[gt_dataset]]
  
  for (sv_caller in names(gr)[names(gr) %in% names_vec])
  {
    if (sv_caller != gt_dataset){
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
}
df <- df[order(df$f1_score, decreasing = TRUE),]
write.table(df, file = "OverlapMillsSVClassify_chr_1_to_3.csv", sep = "\t")

