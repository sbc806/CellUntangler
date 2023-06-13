polar2cart <- function(r, theta) {
  data.frame(x = r * cos(theta), y = r * sin(theta))
}

cart2polar <- function(x, y) {
  data.frame(r = sqrt(x^2 + y^2), theta = atan2(y, x))
}

x = read.delim('/Users/jding/work/tmp_5.csv', 
               header=TRUE, sep=',', row.names = 1)

a = cart2polar(x[, 1], x[, 2])
b = cart2polar(x[, 4], x[, 5])

aa = (2 + 2*cos(a[, 2])) * cos(b[, 2])
bb = (2 + 2*cos(a[, 2])) * sin(b[, 2])
cc = 2 * sin(a[, 2])
             
