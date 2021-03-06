require(StructuralVariantAnnotation)
require(TVTB)
require(S4Vectors)


#SV type assignment based on
# https://github.com/PapenfussLab/gridss/blob/7b1fedfed32af9e03ed5c6863d368a821a4c699f/example/simple-event-annotation.R#L9
apply_svtype <- function(gr)
{
  gr$svtype <-
    ifelse(
      seqnames(gr) != seqnames(partner(gr)),
      "BP",
      ifelse(
        gr$insLen >= abs(gr$svLen) * 0.7,
        "INS",
        ifelse(
          strand(gr) == strand(partner(gr)),
          "INV",
          ifelse(xor(
            start(gr) < start(partner(gr)), strand(gr) == "-"
          ), "DEL",
          "DUP")
        )
      )
    )
  gr
}

# Load the HG002_SVs_Tier1_v0.6.bed confidence regions
load_confidence_regions <- function(sample)
{
  if(sample == 'NA24385')
  {
    bed.file <-
      "/home/cog/smehrem/MinorResearchInternship/BED/HG002_SVs_Tier1_v0.6.bed"
  }else{
    bed.file <-
      "/home/cog/smehrem/MinorResearchInternship/BED/ENCFF001TDO.bed"
  }
  confidence_regions_gr <- rtracklayer::import(bed.file)
  seqlevelsStyle(confidence_regions_gr) <- "NCBI"
  confidence_regions_gr
}

# keep only SVs with both breakpoints overlapping confidence regions
remove_blacklist <- function(gr, confidence_regions_gr, sample)
{
  if(sample == 'NA24385')
  {
    gr[overlapsAny(gr, confidence_regions_gr) &
         overlapsAny(partner(gr), confidence_regions_gr), ]
  }else{
    gr[!(overlapsAny(gr, confidence_regions_gr) |
           overlapsAny(partner(gr), confidence_regions_gr)), ]
  }
  
}


load_sv_caller_vcf <- function(vcf_file, confidence_regions_gr, sample, sv_caller)
{
  # vcf_file <- truth_set_file[[sample]]
  sv_callset_vcf <-
    VariantAnnotation::readVcf(vcf_file)
  
  if(sv_caller =='lumpy')
  {
    # Read evidence support as a proxy for QUAL
    support <- unlist(info(sv_callset_vcf)$SU)
    fixed(sv_callset_vcf)$QUAL <- support
  }else if(sv_caller =='delly')
  {
    # Split-read support plus Paired-end read support as a proxy for QUAL
    sr_support <- info(sv_callset_vcf)$SR
    sr_support[is.na(sr_support)] <- 0
    fixed(sv_callset_vcf)$QUAL <- sr_support + info(sv_callset_vcf)$PE
  }else if (grepl("CNN", sv_caller)){
    # PR score as proxy for QUAL
    infoR <- VcfInfoRules(exprs = list(pr = expression(PR > 0.0)),active = c(TRUE))
    sv_callset_vcf <- S4Vectors::subsetByFilter(sv_callset_vcf, infoR)
    pr_support <- info(sv_callset_vcf)$PR
    fixed(sv_callset_vcf)$QUAL <- pr_support
  }
  
  bpgr <- breakpointRanges(sv_callset_vcf)
  begr <- breakendRanges(sv_callset_vcf)
  gr <- sort(c(bpgr, begr))
  if(sv_caller %in% c('gridss', 'manta')){
    gr <- apply_svtype(gr)
  }
  # Select DEL
  gr <- gr[which(gr$svtype == "DEL")]
  # Select DEL >= 50 bp
  if (grepl("CNN", sv_caller)){
    gr <- gr[gr$svLen>=(50)]
  }else{
  gr <- gr[gr$svLen<=(-50)]
  }
  gr <- remove_blacklist(gr, confidence_regions_gr, sample)
  gr
}

load_truth_set_vcf <- function(vcf_file, confidence_regions_gr, sample)
{
  # vcf_file <- truth_set_file[[sample]]
  sv_callset_vcf <-
    VariantAnnotation::readVcf(vcf_file)
  bpgr <- breakpointRanges(sv_callset_vcf)
  begr <- breakendRanges(sv_callset_vcf)
  gr <- sort(c(bpgr, begr))
  #print(length(gr))
  gr <- gr[which(gr$svtype == "DEL")]
  gr <- gr[gr$svLen<=(-50)]
  #gr <- apply_svtype(gr)
  #print(length(gr))
  # gr <- gr[which(gr$svtype == "DEL")]
  #print(length(gr))
  gr <- remove_blacklist(gr, confidence_regions_gr, sample)
  #print(length(remove_blacklist(gr, confidence_regions_gr, sample)))
  gr
}

load_bedpe <- function(bedpe_file, confidence_regions_gr, sample)
{
  # bedpe_file <- truth_set_file[['NA12878']]
  sv_callset_bedpe <- rtracklayer::import(bedpe_file)
  bpgr <- pairs2breakpointgr(sv_callset_bedpe)
  gr <- sort(bpgr)
  gr <- gr[seqnames(gr) %in% c("1", "2", "3")]
  gr <- remove_blacklist(gr, confidence_regions_gr, sample)
  gr
}

gr <- list()

sample <- 'NA12878'
og_caller <- "gridss"
sv_caller_list <- c('gridss', 'gridssCNN')
negative_set <- c("NoBP_NoBP")
sv_caller_list_new <- c()
truth_set_file <- list()
truth_set_file[['NA12878']] <-
  "/home/cog/smehrem/MinorResearchInternship/VCF/Personalis_1000_Genomes_deduplicated_deletions.bedpe"


gr[[sample]] <- list()
truth_set <- list()

#Load blacklist
confidence_regions_gr <- load_confidence_regions(sample)

if(sample %in% c('NA24385'))
{
  truth_set[[sample]] <- load_truth_set_vcf(truth_set_file[[sample]], confidence_regions_gr, sample)
}else if(sample == 'NA12878'){
  truth_set[[sample]] <- load_bedpe(truth_set_file[[sample]], confidence_regions_gr, sample)
}



# Load sv_callers results
for(sv_caller in sv_caller_list)
{
  print(paste('Loading', sv_caller))
  if (sv_caller == og_caller){
      sv_caller_list_new <- c(sv_caller_list_new, sv_caller)
     # if (sv_caller == 'gridss'){
          vcf_file <- file.path(paste('/home/cog/smehrem/MinorResearchInternship/VCF/FLT/',sv_caller,'.flt.vcf', sep=""))
    #  }else{
         # vcf_file <- file.path(paste('/home/cog/smehrem/MinorResearchInternship/VCF/',sv_caller,'.sym.vcf', sep=""))
     # }
      gr[[sample]][[sv_caller]] <-
        load_sv_caller_vcf(vcf_file, confidence_regions_gr, sample, sv_caller)
  }else{
    for (neg in negative_set){
        for (i in 1:10){
          sv_caller_name <- paste(sv_caller,neg,i, sep='_')
          sv_caller_list_new <- c(sv_caller_list_new, sv_caller_name)
          vcf_file <- file.path('/home/cog/smehrem/breakpoint-pairs/ConsensusCNN_VCF',neg,paste("FULLCalledSV_",og_caller,"_",i,".vcf",sep=''))
          gr[[sample]][[sv_caller_name]] <-
            load_sv_caller_vcf(vcf_file, confidence_regions_gr, sample, sv_caller)
          #print(load_sv_caller_vcf(vcf_file, confidence_regions_gr, sample, sv_caller))
        }
    }
      
  }
}

truth_svgr <- truth_set[[sample]]
for(sv_call in sv_caller_list_new)
{
  gr[[sample]][[sv_call]]$caller <- sv_call
  temp_gr <- gr[[sample]][[sv_call]]
  temp_gr <- temp_gr[seqnames(temp_gr) %in% c("1", "2", "3")]
  gr[[sample]][[sv_call]] <- temp_gr
  
}


for(svcaller in sv_caller_list_new)
{
  if (sv_caller == og_caller){
  gr[[sample]][[svcaller]]$truth_matches <- countBreakpointOverlaps(gr[[sample]][[svcaller]], truth_svgr,
                                                                    # read pair based callers make imprecise calls.
                                                                    # A margin around the call position is required when matching with the truth set
                                                                    maxgap=200,
                                                                    # Since we added a maxgap, we also need to restrict the mismatch between the
                                                                    # size of the events. We don't want to match a 100bp deletion with a 
                                                                    # 5bp duplication. This will happen if we have a 100bp margin but don't also
                                                                    # require an approximate size match as well
                                                                    sizemargin=0.25,
                                                                    ignore.strand = TRUE,
                                                                    # We also don't want to match a 20bp deletion with a 20bp deletion 80bp away
                                                                    # by restricting the margin based on the size of the event, we can make sure
                                                                    # that simple events actually do overlap
                                                                    restrictMarginToSizeMultiple=0.5,
                                                                    # Some callers make duplicate calls and will sometimes report a variant multiple
                                                                    # times with slightly different bounds. countOnlyBest prevents these being
                                                                    # double-counted as multiple true positives.
                                                                    countOnlyBest=TRUE)
  }else{
    gr[[sample]][[svcaller]]$truth_matches <- countBreakpointOverlaps(gr[[sample]][[svcaller]], truth_svgr,
                                                                      # read pair based callers make imprecise calls.
                                                                      # A margin around the call position is required when matching with the truth set
                                                                      maxgap=200,
                                                                      # Since we added a maxgap, we also need to restrict the mismatch between the
                                                                      # size of the events. We don't want to match a 100bp deletion with a 
                                                                      # 5bp duplication. This will happen if we have a 100bp margin but don't also
                                                                      # require an approximate size match as well
                                                                      sizemargin=0.25,
                                                                      ignore.strand = TRUE,
                                                                      # We also don't want to match a 20bp deletion with a 20bp deletion 80bp away
                                                                      # by restricting the margin based on the size of the event, we can make sure
                                                                      # that simple events actually do overlap
                                                                      restrictMarginToSizeMultiple=0.5)
    # Some callers make duplicate calls and will sometimes report a variant multiple
    # times with slightly different bounds. countOnlyBest prevents these being
    # double-counted as multiple true positives.
    #countOnlyBest=TRUE)
  }
}
svgr <- data.frame()

for (sv_caller in sv_caller_list_new){
  svgr <- rbind(svgr, as.data.frame(gr[[sample]][[sv_caller]]))
}


test <- svgr %>%
  dplyr::select(QUAL, caller, truth_matches) %>%
  dplyr::group_by(caller) %>%
  dplyr::summarise(
    calls=n(),
    True_Positives=sum(truth_matches > 0)) %>%
  dplyr::group_by(caller) %>%
  #dplyr::arrange(dplyr::desc(QUAL)) %>%
  dplyr::mutate(
    Positives=cumsum(calls),
    True_Positives=cumsum(True_Positives),
    False_Positives=Positives - True_Positives,
    Precision=True_Positives / Positives,
    Recall=True_Positives /length(truth_svgr), 
    F1_Score=2*(Precision*Recall/(Precision+Recall)))
drops <- c("Positives")
performance_df <- test[ , !(names(test) %in% drops)]
write.csv(performance_df,"/home/cog/smehrem/breakpoint-pairs/RScript/PerformanceBenchmarkGridssCNNGridss_NoBP_NoBP.csv", sep=",")
# Plotting Precision and Recall, from StructuralVariantAnnotation vignette:
# https://bioconductor.org/packages/devel/bioc/vignettes/StructuralVariantAnnotation/inst/doc/vignettes.html

# main.title <- c("NA12878\nsv_classify truth set")
# names(main.title) <- c("NA12878")
# 
# suppressPackageStartupMessages(require(dplyr))
# suppressPackageStartupMessages(require(ggplot2))

# ggplot(svgr %>%
#          dplyr::select(QUAL, caller, truth_matches) %>%
#          dplyr::group_by(caller, QUAL) %>%
#          dplyr::summarise(
#            calls=n(),
#            tp=sum(truth_matches > 0)) %>%
#          dplyr::group_by(caller) %>%
#          dplyr::arrange(dplyr::desc(QUAL)) %>%
#          dplyr::mutate(
#            cum_tp=cumsum(tp),
#            cum_n=cumsum(calls),
#            cum_fp=cum_n - cum_tp,
#            precision=cum_tp / cum_n,
#            recall=cum_tp/length(truth_svgr))) +
#   aes(x=recall, y=precision, colour=caller) +
#   geom_point(size=0.2) +
#   theme(text = element_text(size=30))+
#   geom_line() +
#   scale_y_continuous(labels=scales::percent, limit=c(0,1)) +
#   scale_x_continuous(
#     labels=scales::percent,
#     sec.axis = sec_axis(~(.)*length(truth_svgr), name = "true positives")) 
#   #labs(title=main.title[sample])
# ggsave("../figures/FULLGridssCNNbenchmark_TrueSV_TrueSV.png", plot = last_plot(), scale = 2, dpi = 200, height=7, width=11)

