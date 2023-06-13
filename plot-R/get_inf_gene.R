# https://davetang.org/muse/2011/05/20/extract-gene-names-according-to-go-terms/

library(org.Hs.eg.db)
library(GO.db)

# GO:0060337
go_id = GOID( GOTERM[ Term(GOTERM) == "type I interferon signaling pathway"])

allegs = get(go_id, org.Hs.egGO2ALLEGS)

# 95 genes
genes = unique(unlist(mget(allegs,org.Hs.egSYMBOL)))
names(genes) = NULL


# mouse
library('org.Mm.eg.db')
go_id = GOID( GOTERM[ Term(GOTERM) == "type I interferon signaling pathway"])

allegs = get(go_id, org.Mm.egGO2ALLEGS)

# 39 genes
gene.mouse = unique(unlist(mget(allegs, org.Mm.egSYMBOL)))

names(gene.mouse) = NULL



# ======
go_id = GOID( GOTERM[ Term(GOTERM) == "response to hypoxia"])

allegs = get(go_id, org.Hs.egGO2ALLEGS)

# 359 genes
gene.hypoxia = unique(unlist(mget(allegs,org.Hs.egSYMBOL)))
names(gene.hypoxia) = NULL



# ======
go_id = GOID( GOTERM[ Term(GOTERM) == "cell cycle"])

allegs = get(go_id, org.Hs.egGO2ALLEGS)

# 359 genes
gene.hypoxia = unique(unlist(mget(allegs,org.Hs.egSYMBOL)))
names(gene.hypoxia) = NULL




library(org.Hs.eg.db)
library(GO.db)

# GO:0060337
go_id = GOID( GOTERM[ Term(GOTERM) == "stress"])

allegs = get(go_id, org.Hs.egGO2ALLEGS)

# 95 genes
genes = unique(unlist(mget(allegs,org.Hs.egSYMBOL)))
names(genes) = NULL



