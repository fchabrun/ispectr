rm(list=ls())

source("R/it_functions.r")

library(openxlsx)
library(stringr)
library(tidyverse)
library(gridExtra)


db_data = read.xlsx("C:/Users/flori/Documents/Home/Research/SPECTR/ISPECTR/data/cameron_may_2024/is_data_xlsx/GSH Old MiniCap Immuno 2017.xlsx")

dim(db_data)  # 1153 x 128

# limit to F program (IS) and J (SPEP)
db_data <- db_data[db_data$programma %in% c("F", "J"), ]

strsplits <- str_split(db_data$id, "-")

db_data$id_unique <- sapply(strsplits, function(x) {x[1]})
db_data$id_track <- sapply(strsplits, function(x) {ifelse(is.na(x[2]),"",x[2])})
db_data[ ,"id_track"] <- ifelse(db_data$programma == "J", "Ref", db_data[ ,"id_track"])
db_data$id_ctrl <- substr(db_data$id_unique, 1, 9) == "CONTROLID"
db_data <- db_data[!db_data$id_ctrl, ]

#check how many, etc.
db_data <- db_data %>%
  group_by(id_unique) %>%
  mutate(n_ref = sum(id_track == "Ref")) %>%
  mutate(n_elp = sum(id_track == "ELP")) %>%
  mutate(n_g = sum(id_track == "IgG")) %>%
  mutate(n_a = sum(id_track == "IgA")) %>%
  mutate(n_m = sum(id_track == "IgM")) %>%
  mutate(n_k = sum(id_track == "K")) %>%
  mutate(n_l = sum(id_track == "L"))

db_data$checked_valid = (db_data$n_ref == 1) & (db_data$n_elp == 1) & (db_data$n_g == 1) & (db_data$n_a == 1) & (db_data$n_m == 1) & (db_data$n_k == 1) & (db_data$n_l == 1)

table(db_data$checked_valid)

# filter the db
db_data <- db_data[sample_valid, ]

# get an example sample and plot it
# tmp = decipherCurve_v2024_2(db_data[1, ], italiano=T)
# drawProcessedSPE(tmp$curve)

for (unique_id in unique(db_data$id_unique)) {
  # for each single patient
  sample_subset = db_data[db_data$id_unique == unique_id, ]
  sample_data = list()
  # parse the different IF tracks
  for (if_track in c("ELP", "IgG", "IgA", "IgM", "K", "L")) {
    sample_track_subset = sample_subset[sample_subset$id_track == if_track, ]
    # check exactly 1 XXX track for patient YYY
    if (nrow(sample_track_subset) != 1) {
      stop("Unexpected number of rows")
    }
    decrypted_data = decipherCurve_v2024_2(sample_track_subset, italiano=T)
    # check no error/warning
    if (decrypted_data$alertflag > 0) {
      warning(paste0("Alert flag found at ", decrypted_data$alertflag, " for sample ", unique_id, " at track ", if_track))
    }
    # save
    sample_data[[if_track]] <- decrypted_data$curve
  }
  # plot
  gps = list()
  for (if_track in c("ELP", "IgG", "IgA", "IgM", "K", "L")) {
    new_gp <- drawProcessedSPE(sample_data[[if_track]])
    gps[[length(gps) + 1]] <- new_gp
  }
  grid.arrange(grobs = gps, nrow = 6)
  # skip to next
  break
}

# TODO plus qu'à exporter vers python
# TODO indice: on voit que le pic disparaît de la piste pos, mais on voit aussi que la précipitation crée une bosse vers l'albu
# => on pourrait trouver un moyen de "masquer" le pic pour forcer le modèle à regarder pas seulement le pic mais aussi l'albu de cette manière! (plus robuste?)





































