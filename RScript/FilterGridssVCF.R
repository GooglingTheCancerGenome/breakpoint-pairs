require(StructuralVariantAnnotation)
require(TVTB)
require(S4Vectors)
require(rtracklayer)

file_gridss <- "/home/cog/smehrem/MinorResearchInternship/VCF/FLT/gridss.flt.vcf"
sv_callset_vcf <-
  VariantAnnotation::readVcf(file_gridss)

bpgr <- breakpointRanges(sv_callset_vcf) 
begr <- breakendRanges(sv_callset_vcf)
gr_gridss <- sort(c(bpgr, begr))

simpleEventType <- function(gr) {
  return(ifelse(seqnames(gr) != seqnames(partner(gr)), "ITX", # inter-chromosomosal
                ifelse(gr$insLen >= abs(gr$svLen) * 0.7, "INS",
                       ifelse(strand(gr) == strand(partner(gr)), "INV",
                              ifelse(xor(start(gr) < start(partner(gr)), strand(gr) == "-"), "DEL", "DUP")))))
}

gr_gridss$svtypes <- simpleEventType(gr_gridss)
gr_dels <- gr_gridss[gr_gridss$svtypes == "DEL"]
write.table(gr_dels$sourceId,"/home/cog/smehrem/MinorResearchInternship/VCF/FLT/gridss_ID_DEL.txt", sep="\n", row.names = FALSE, col.names = FALSE,quote = FALSE )