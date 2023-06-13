# load MNIST data
# https://gist.github.com/brendano/39760

load_label_file <- function(filename) {
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  n = readBin(f,'integer',n=1,size=4,endian='big')
  y = readBin(f,'integer',n=n,size=1,signed=F)
  close(f)
  y
}

filename = '/Users/jding/work/mvae/data/MNIST/raw/t10k-labels-idx1-ubyte'
a = load_label_file(filename)

xx = read.delim('/Users/jding/work/tmp1.csv', sep=',', row.names = 1)

PlotCycle(xx, a)

# =====
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

open3d()

plot3d(xx[, 1:3], col=cbPalette[10])
cylinder3d(len=7, rad=2, col = "lightgray", n = 100)
decorate3d() 

rgl.postscript('~/work/jiarui-ding/research-statement/figure/structure/cycle/cell_cycle.pdf',
               fmt = "pdf")



# ================================================================
x3 = xx

x = seq(-20, 20, length = 10)
y = seq(-50, 50, length = 10)
f = function(x, y) { r <- sqrt(x^2 + y^2 + 1)}

z <- outer(x, y, f)
z[is.na(z)] <- 1


open3d()
bg3d("white")
material3d(col = "black")

par3d(windowRect = c(5, 50, 1280, 1024))

plot3d(x=x3[,2], y=x3[,3], z=x3[,1] - 0.01, add=FALSE, 
       col=AssignLabelColor(cc, meta$embryo.time.bin, uniq.label),
       xlab='z1', ylab='z2', zlab='z0')

surface3d(x, y, z, col = "lightgray",
          xlab = "Z", ylab = "Y", zlab = "X", 
          polygon_offset = 0, alpha=0.5)

persp3d(x, y, z, front = "lines", back = "lines", 
        lit = FALSE, add = TRUE, alpha=0.5)





PlotCycle = function(x, cluster, col, density=FALSE, 
         legend=TRUE, cex.lab=1.8, 
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
  plot3d(x[, 1:3], col = col.point,
         # xlim=c(-1, 1), ylim=c(-1, 1), zlim=c(-1, 1),
         # box=FALSE, axes=FALSE, xlab='', ylab='', zlab=''
         )
  
  # arrow3d(c(0, -1.32, 0), c(0, 1.32, 0),
  #         col = 'gray', s=0.04, type='extrusion', lit=FALSE)
  # 
  # # arrow3d(c(1.32, 0, 0), c(-1.32, 0, 0),
  # #         col = 'gray', s=0.04, type='extrusion', lit=FALSE)
  # 
  # # col https://kbroman.wordpress.com/2014/05/07/crayon-colors-in-r/
  # spheres3d(0, 0, 0, lit=FALSE, color='#dbd7d2',  # brocolors("crayons")['Timberwolf'],
  #           alpha=0.95, radius=0.99)
  # spheres3d(0, 0, 0, radius=0.9999, lit=FALSE, color='gray',
  #           front='lines', alpha=0.95)
  
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
      
      k.centers[, 1:2] = k.centers[, 1:2] / sqrt(rowSums(k.centers[, 1:2]^2)) * 2.15
      
      #id = which(cluster.srt %in% c('MT-hi', 'CD69- Mast'))
      #cluster.srt[id] = ''
      
      text3d(k.centers, texts=cluster.srt, col='black', cex=cex.lab)
    }
  } 
  # else {
  #   leg = quantile(cluster, probs = c(0, 0.25, 0.5, 0.75, 1))
  #   
  #   id = rep(NA, 5)
  #   
  #   for (i in seq(5)) {
  #     ii = which.min(abs(cluster - leg[i]))
  #     id[i] = ii
  #     leg[i] = cluster[ii]
  #   }
  #   
  #   col.leg = col.point[id]
  #   leg = cluster[id]
  #   names(col.leg) = leg
  #   
  #   if (legend == TRUE) {
  #     legend3d(leg.pos, legend = leg,
  #              pch = 16, col = col.leg, 
  #              cex=cex.leg, inset=inset.leg, bty='n')
  #   } else {
  #     legend3d(leg.pos, legend = '',
  #              pch = 16, col = 'white', cex=1, 
  #              inset=c(0.02), box.lwd=0, bty='n')
  #   }
  # }
}


