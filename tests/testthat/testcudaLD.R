context("Make sure that LD calculation works")

test_that("LD shrinkage estimators work as expected on simulated data",{
  m <- 100
  Ne <- 10000
  n <- 100
  p <- 5000
  cutoff <- 1e-3
  tmap <- cumsum(runif(p)/10)
  
  # library(RcppEigenH5) 
  #  ikgf <- "/media/nwknoblauch/Data/GTEx/GTEx_rssr/Genome_SNP/SNP_Whole_Blood_1kg_6250_t.h5"
  #  cuda_f <- "/home/nwknoblauch/Dropbox/test_cuda/test_cov.h5"
  #  cuda_ijk <- read_df_h5(cuda_f,"R",subcols=c("k","j","i"))
  #  cuda_sighat <- read_2d_mat_h5(cuda_f,"R","LDshrink")
  #  cuda_cor <- cov2cor(cuda_sighat)
  #  matA <- read_2d_index_h5(ikgf,"SNPdata","genotype",1:6000)
  #  mapA <- read_dvec(ikgf,"SNPinfo","map")[1:6000]
  # m=85
  # Ne=1490.672741
  # cutoff=1e-3
  # nLD <- calcLD(matA,mapA,m,Ne,cutoff)
  
  
  #   
  
  Hpanel <- matrix(sample(c(0,1),n*2*p,replace=T),n*2,p)
  # mfile <- system.file("m_files/run_install.m",package="rssr")
  # mdir <- system.file("m_files",package="RSSReQTL")
  
  #change to the directory with the .m files in Octave
  
  system.time(oRsig <- LDshrink::calcLD(hmata = Hpanel,mapa = tmap,m = m,Ne = Ne,cutoff = cutoff))
  system.time(Rsig <- calcLD_CUDA(hmata = Hpanel,map = tmap,m = m,Ne = Ne,cutoff = cutoff))
  
  
  
  # Rsig[lower.tri(Rsig)] <- 0
  testthat::expect_equal(Rsig,oRsig,tolerance=1e-5)
})



test_that("sparse LD shrinkage estimators generate the correct sparse matrices",{
  m <- 100
  Ne <- 10000
  n <- 100
  p <- 5000
  cutoff <- 1e-3
  tmap <- cumsum(runif(p)/10)
  
  # library(RcppEigenH5) 
  #  ikgf <- "/media/nwknoblauch/Data/GTEx/GTEx_rssr/Genome_SNP/SNP_Whole_Blood_1kg_6250_t.h5"
  #  cuda_f <- "/home/nwknoblauch/Dropbox/test_cuda/test_cov.h5"
  #  cuda_ijk <- read_df_h5(cuda_f,"R",subcols=c("k","j","i"))
  #  cuda_sighat <- read_2d_mat_h5(cuda_f,"R","LDshrink")
  #  cuda_cor <- cov2cor(cuda_sighat)
  #  matA <- read_2d_index_h5(ikgf,"SNPdata","genotype",1:6000)
  #  mapA <- read_dvec(ikgf,"SNPinfo","map")[1:6000]
  # m=85
  # Ne=1490.672741
  # cutoff=1e-3
  # nLD <- calcLD(matA,mapA,m,Ne,cutoff)
  
  
  #   
  
  Hpanel <- matrix(sample(c(0,1),n*2*p,replace=T),n*2,p)
  # mfile <- system.file("m_files/run_install.m",package="rssr")
  # mdir <- system.file("m_files",package="RSSReQTL")
  
  #change to the directory with the .m files in Octave
  system.time(Rsig <- calcLD_CUDA(hmata = Hpanel,map = tmap,m = m,Ne = Ne,cutoff = cutoff))
  system.time(sparseSig <- calcLD_CUDA_sp(hmata = Hpanel,map = tmap,m = m,Ne = Ne,cutoff = cutoff))
  
  sm <- sparseMatrix(i=sparseSig$outerIndex,p = sparseSig$innerIndices,x=sparseSig$values,dims=c(p,p),index1 = F)
  Rm <- as(Rsig,"sparseMatrix")
  
  # Rsig[lower.tri(Rsig)] <- 0
  testthat::expect_equal(sm,Rm)
})
