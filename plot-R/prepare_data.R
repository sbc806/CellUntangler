# x = readRDS('/Users/jding/work/desc/data/eset-final.rds')
# 
# theta = x@phenoData$theta
# 
# source('~/util/scphere_util.R')
# 
# gene = featureData(x)$name
# 
# dd = Matrix(exprs(x))
# rownames(dd) = gene
# 
# cycle.gene= unique(unlist(RevelioGeneList))
# gene = intersect(cycle.gene, gene)
# obj = ClusterSeurat(dd[gene, ])
# 
# obj@meta.data$theta = theta
# FeaturePlot(obj, 'theta')
#
#
# x = read.delim('/Users/jding/Downloads/GSE142356_RAW/GSM4226257_out_gene_exon_tagged.dge_exonsds_046.txt',
#                 row.names = 1)
# 
# obj = ClusterSeurat(x)



library(Revelio)
myData <- createRevelioObject(rawData = revelioTestData_rawDataMatrix,
                              cyclicGenes = revelioTestData_cyclicGenes)


myData <- getCellCyclePhaseAssignInformation(dataList = myData)

myData <- getPCAData(dataList = myData)
myData <- getOptimalRotation(dataList = myData)

myData <- getPCAData(dataList = myData, boolPlotResults = TRUE)
myData <- getOptimalRotation(dataList = myData, boolPlotResults = TRUE)

source('~/util/util.R')
# 1477 cells
ScatterLabel(t(myData@transformedData$dc$data[1:2, ]), as.character(myData@cellInfo$ccPhase))

# =====
# y = read.delim('/Users/jding/work/mvae/chkpt/vae-mnist-s1,e1-2021-04-20T23:06:03.449857/repr/e1.txt', 
#                header=FALSE, sep=' ')
# x = read.delim('/Users/jding/work/mvae/chkpt/vae-mnist-s1,e1-2021-04-20T20:53:48.811989/repr/s1.txt', 
#                header=FALSE, sep=' ')
# 
# xx = cbind(x, y)
xx = read.delim('/Users/jding/work/mvae/chkpt/vae-mnist-s1,e1-2021-04-20T23:35:18.863095/repr/all.txt',
                header=FALSE, sep=' ')

xx = read.delim('/Users/jding/work/mvae/all_encode_v23.txt',
                header=FALSE, sep=' ')

PlotCycle(xx, as.character(myData@cellInfo$ccPhase), rad = 1, len = 5)


ScatterLabel(xx[, c(1,2)], as.character(myData@cellInfo$ccPhase), 
             pch=1:5, cex = 1, show.legend = TRUE, 
             xpd = TRUE, ncol = 5, inset=c(0, -0.09))


# plot3d(xx, col=AssignLabelColor(1:10, as.character(myData@cellInfo$ccPhase)))
# 
# 
open3d()

plot3d(xx[, 1:3], col=AssignLabelColor(cbPalette, as.character(myData@cellInfo$ccPhase)))
cylinder3d(len=2, rad=0.25, col = "lightgray", n = 100)
decorate3d()






PlotCycle = function(x, cluster, col, density=FALSE, 
                     legend=TRUE, cex.lab=1.8, 
                     rad=0.25, len=5, 
                     cex.leg=2, inset.leg=c(0.1, 0,1), 
                     show.lab=TRUE, 
                     leg.pos='topright', sort.cluster=TRUE) {
  
  # par3d(windowRect = c(100, 100, 1000, 1000))
  uniq.cluster = unique(cluster)
  if (sort.cluster == TRUE) {
    uniq.cluster = sort(uniq.cluster)
  }
  
  
  if (missing(col)) {
    col = distinct.col
  }
  if (density == FALSE) {
    col.point = AssignLabelColor(col, cluster,
                                 uniq.label=uniq.cluster)
    # col.point = AssignLabelColor(col, cluster)
  } else {
    colours = colorRampPalette((brewer.pal(7, "YlOrRd")))(10)
    FUN = colorRamp(colours)
    
    cluster = (cluster - min(cluster)) / diff(range(cluster))
    col.point = rgb(FUN(cluster), maxColorValue=256)
  }
  # plot3d(x[, 1:3], col = col.point,
  #        # xlim=c(-1, 1), ylim=c(-1, 1), zlim=c(-1, 1),
  #        # box=FALSE, axes=FALSE, xlab='', ylab='', zlab=''
  # )
  
  # open3d()
  
  plot3d(x[, 1:3], col=col.point)
  cylinder3d(len=len, rad=rad, col = "lightgray", n = 100)
  decorate3d() 
  
  
  if (density == FALSE) {
    id = !duplicated(cluster)
    col.leg = col.point[id]
    leg = cluster[id]
    names(col.leg) = leg
    
    if (legend == TRUE) {
      legend3d(leg.pos, legend = leg,
               pch = 16, col = col.leg, 
               cex=cex.leg, inset=inset.leg, bty='n')
    } else {
      legend3d(leg.pos, legend = '',
               pch = 16, col = 'white', cex=1, 
               inset=c(0.02), box.lwd=0, bty='n')
    }
    
    if (show.lab == TRUE) {
      cluster.srt = sort(unique(cluster))
      k.centers = sapply(cluster.srt, function(zz) {
        cat(zz, '\t')
        id = cluster == zz
        center = colMedians(as.matrix(x[id, , drop=FALSE]))
      })
      
      k.centers = t(k.centers)
      
      cluster.size = table(cluster)[as.character(cluster.srt)]
      id = which(cluster.size > 0)
      
      if (length(id) > 0) {
        k.centers = k.centers[id, , drop=FALSE]
        cluster.srt = cluster.srt[id]
      }
      
      k.centers[, 1:2] = k.centers[, 1:2] / sqrt(rowSums(k.centers[, 1:2]^2)) * (rad + 0.2*rad)
      
      #id = which(cluster.srt %in% c('MT-hi', 'CD69- Mast'))
      #cluster.srt[id] = ''
      
      text3d(k.centers, texts=cluster.srt, col='black', cex=cex.lab)
    }
  } 
}


cylinder3d <- function(len=1, rad=1, n=30, ctr=c(0,0,0),
                       trans = par3d("userMatrix"),...) {
  if (missing(trans) && !rgl.cur()) trans <- diag(4)
  degvec <- seq(0,2*pi,length=n)
  circ <- cbind(rad*cos(degvec),rad*sin(degvec))
  x <- rad*cos(degvec)
  y <- rad*sin(degvec)
  p <- c(1,1,rep(2:n,each=4),1,1)
  z <- rep(c(-1,1,1,-1)*len/2,n)
  quads3d(x[p], y[p], z,...)
}



# =====
source('~/util/scphere_util.R')

# 1564 cells
x = Matrix(as.matrix(revelioTestData_rawDataMatrix))

cell = myData@cellInfo$cellID
x = x[, cell]
save(x, file='~/work/mvae/data/x.rda')

obj = ClusterSeurat(x)

UMAPPlot(obj, group.by='orig.ident')

obj@meta.data[, 'cell_cycle'] = myData@cellInfo$ccPhase

UMAPPlot(obj, group.by='cell_cycle')

DimPlot(obj, group.by='cell_cycle', reduction = 'pca')
DimPlot(obj, group.by='orig.ident', reduction = 'pca')

# =====
writeMM(x, file = '/Users/jding/work/desc/data/hela_1477.mtx')
write.table(rownames(x), file='/Users/jding/work/desc/data/hela_gene.txt', 
            row.names=FALSE, quote=FALSE, col.names = FALSE)

write.table(colnames(x), file='/Users/jding/work/desc/data/hela_cell.txt', 
            row.names=FALSE, quote=FALSE, col.names = FALSE)


gene.cycle = unique(unlist(revelioTestData_cyclicGenes))

write.table(gene.cycle, file='/Users/jding/work/desc/data/cycle_gene.txt', 
            row.names=FALSE, quote=FALSE, col.names = FALSE)

ClusterSeurat1 = function(x, species='human', resolution=0.8, 
         nfeatures=2000, ndim=25, 
         gene.filt=NULL, marker.gene=NULL, 
         scale.factor=10000, ...) {
  obj = CreateSeuratObject(counts = x)
  
  if (species == 'human') {
    obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")
    gene.rm = grep(rownames(obj@assays$RNA@counts), 
                   pattern = '^RPS|^RPL|^MRPS|^MRPL|MT-', value = TRUE)
  } else {
    obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^mt-")
    gene.rm = grep(rownames(obj@assays$RNA@counts), 
                   pattern = '^Rps|^Rpl|^Mrps|^Mrpl|mt-', value = TRUE)
  }
  
  obj = NormalizeData(object = obj)
  obj = FindVariableFeatures(object = obj, nfeatures=nfeatures)
  var.gene = g 
  
  # obj = ScaleData(object = obj, vars.to.regress = "nFeature_RNA")
  obj = ScaleData(object = obj)
  
  obj = RunPCA(object = obj)
  
  obj
}

# Variable genes and also normalization has a significant impact


x = x[intersect(gene.cycle, rownames(x)), ]
obj = ClusterSeurat1(x)
ScatterLabel(obj@reductions$pca@cell.embeddings[, 1:2], as.character(myData@cellInfo$ccPhase))

ComputePC = function(x, n=50) {
  # x = ScaleSparseMatrix(x) * 10000
  x = TransformSparseMatrix(x, 'log2')
  
  x = irlba::prcomp_irlba(t(x), n = n, center = TRUE, scale=FALSE)
  
  x = x$x[, 1:n]
  
  x
}


ComputePC = function(x, n=50) {
  # x = ScaleSparseMatrix(x) * 10000
  x = TransformSparseMatrix(x, 'log2')
  x = t(x)
  
  x  = t(scale(t(x), center=FALSE, scale=TRUE))
  
  x = irlba::prcomp_irlba(x, n = n, center = TRUE, scale=FALSE)
  
  x = x$x[, 1:n]
  
  x
}



pc = ComputePC(x)
ScatterLabel(pc[, 1:2], as.character(myData@cellInfo$ccPhase))
source('~/util/scphere_util.R')

x = read.delim('~/work/mvae/exp_matrix.tsv', header=FALSE, sep=' ')
x = Matrix(as.matrix(x)[, 1:714])
x = t(x)

# =====
x = read.delim('~/work/mvae/exp_matrix.tsv', header=FALSE, sep=' ')
x = Matrix(as.matrix(x)[, 715:ncol(x)])
x = t(x)

pc = ComputePC(x)
ScatterLabel(pc[, 1:2], as.character(myData@cellInfo$ccPhase))
source('~/util/scphere_util.R')

colnames(x) = seq(ncol(x))
a1 = ClusterSeurat(x)

ScatterLabel(a1@reductions$umap@cell.embeddings[, 1:2], as.character(myData@cellInfo$ccPhase))


# =====
cart2polar <- function(x, y) {
  data.frame(r = sqrt(x^2 + y^2), theta = atan2(y, x))
}

# 25
xx = read.delim('/Users/jding/work/mvae/all_encode_v33.txt',
                header=FALSE, sep=' ')

a = cart2polar(xx[, 1], xx[, 2])
b = cart2polar(xx[, 3], xx[, 4])

aa = (2 + 2*cos(a[, 2])) * cos(b[, 2])
bb = (2 + 2*cos(a[, 2])) * sin(b[, 2])
cc = 2 * sin(a[, 2])

plot3d(aa, bb, cc, col=AssignLabelColor(distinct.col, as.character(myData@cellInfo$ccPhase)))

PlotSphere(xx[, 3:5], as.character(myData@cellInfo$ccPhase))

# A torus seems to work 
ScatterLabel(xx[, c(1:2)], as.character(myData@cellInfo$ccPhase), 
             pch=1:5, cex = 0.6, show.legend = TRUE, 
             xpd = TRUE, ncol = 5, inset=c(0, -0.09))

ScatterLabel(xx[, c(3:4)], as.character(myData@cellInfo$ccPhase), 
             pch=1:5, cex = 0.6, show.legend = TRUE, 
             xpd = TRUE, ncol = 5, inset=c(0, -0.09))


# OK, S1,S10, S1,H2
# S1,S2 

# # ‘mapproj’
a = car2sph(xx[, c(1,3,2)])

aa = mapproject(a[,2], a[,1], projection = 'gilbert')

ScatterLabel(cbind(aa$x, aa$y), as.character(myData@cellInfo$ccPhase))

plot3d(obj@reductions$pca@cell.embeddings[, 1:3], 
       col=AssignLabelColor(distinct.col, as.character(myData@cellInfo$ccPhase)))

# Assign to cell cycle stages





