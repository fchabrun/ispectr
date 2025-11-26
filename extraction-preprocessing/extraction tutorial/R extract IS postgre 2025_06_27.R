# HERE: REPLACE THE PATH BY THE PATH TO THE .dump FILE
working_directory = "C:/Users/flori/OneDrive - univ-angers.fr/Documents/Home/Research/SPECTR/ISPECTR/data/lemans_2025"

# INSTALL THE "data.table" PACKAGE IF NOT ALREADY INSTALLED, AND LOAD IT
if (!("data.table" %in% installed.packages())) {
  install.packages("data.table")
}
# THIS PACKAGE WILL BE USED TO QUICKLY LOAD A DATA FILE (THE SQL DUMP IS EXPECTED TO BE SEVERAL GB)
library(data.table)
# SAME FOR PACKAGE STRINGR: WILL ALLOW TO EASILY MANIPULATE STRINGS
if (!("stringr" %in% installed.packages())) {
  install.packages("stringr")
}
library(stringr)
# SAME FOR PACKAGE DPLYR (DF MANIPULATION)
if (!("dplyr" %in% installed.packages())) {
  install.packages("dplyr")
}
library(dplyr)
# SAME FOR PACKAGE DIGEST: WILL ALLOW TO CREATE RANDOM IDENTIFIERS (MD5)
if (!("digest" %in% installed.packages())) {
  install.packages("digest")
}
library(digest)
# FINALLY HDF TO CREATE H5 DATASETS
if (!("rhdf5" %in% installed.packages())) {
  install.packages("BiocManager")
  BiocManager::install("rhdf5")
}
library(rhdf5)

# WE DEFINE A QUICK FUNCTION THAT WILL QUICKLY PARSE THE DUMP TO FIND THE HEADER
# SO WE WILL KNOW AT WHICH LINE THE DATA STARTS, AND ALSO THE NAMES OF THE COLUMNS
read_file_header = function(filepath, max_read_lines=100) {
  con = file(filepath, "r")
  n <- 0
  while ( TRUE ) {
    n = n + 1
    if (n > max_read_lines) {
      close(con)
      stop("Too many lines read before encountering header")
    }
    line = readLines(con, n = 1)
    if ( length(line) == 0 ) {
      close(con)
      stop("End of the file reached before encountering header")
    }
    if ((nchar(line) > 15) && (substr(line, 1, 15) == "COPY anagrafica")) {
      close(con)
      if (!grepl("^COPY anagrafica [(](.+)[)].+$", line)) {
        stop(paste0("Invalid header: ", line))
      }
      header_columns = strsplit(gsub("^COPY anagrafica [(](.+)[)].+$", "\\1", line), ", ")[[1]]
      return(list("header_position"=n, "header_columns"=header_columns))
    }
  }
  close(con)
  stop("End of the file reached before encountering header")
}

# MAKE THE NAME OF THE DUMP FILE THAT WILL BE LOADED
path_to_dump = file.path(working_directory, "extract.dump")

# PARSE THE FILE A FIRST TIME AND LOOK FOR THE HEADER
header_data <- read_file_header(filepath=path_to_dump)
print(paste0("Found header data at line ", header_data$header_position, " with ", length(header_data$header_columns), " columns"))

# THEN PARSE THE FILE A SECOND TIME TO LOAD THE DATA
raw_dataset = as.data.frame(fread(path_to_dump, sep="\t", skip=header_data$header_position))
# AND ADJUST COLUMN NAMES ACCORDING TO THE PARSED HEADER
colnames(raw_dataset) <- header_data$header_columns
print(paste0("Found data for ", nrow(raw_dataset), " total rows"))

# DISCARD UNUSED COLUMNS, AND RENAME THEM
# DEFINE COLUMNS WE WANT TO KEEP
cols_keep <- c("data_nascita", "programma", "data_prel", "nominativo", "data_analisi",
               "id", "pt", "sesso", "reparto",
               "commento1", "longmemo", "numfraz",
               "fraz_1", "fraz_2", "fraz_3", "fraz_4", "fraz_5", "fraz_6", "fraz_7", "fraz_8", "fraz_9", "fraz_10",
               "nome_1", "nome_2", "nome_3", "nome_4", "nome_5", "nome_6", "nome_7", "nome_8", "nome_9", "nome_10",
               "curva", "originalcurve",
               "val_picco_1", "val_picco_2", "val_picco_3", "val_picco_4",
               "nome_picco_1", "nome_picco_2", "nome_picco_3", "nome_picco_4",
               "coord_picco")
# DEFINE NEW COLUMN NAMES
cols_rename <- c("birth_date", "program", "sampling_date", "fullname", "analysis_date",
                 "analysis_id", "total_protein", "sex", "uf_code",
                 "comment_1", "comment_long", "number_fractions",
                 "fraction1_percent", "fraction2_percent", "fraction3_percent", "fraction4_percent", "fraction5_percent", "fraction6_percent", "fraction7_percent", "fraction8_percent", "fraction9_percent", "fraction10_percent",
                 "fraction1_name", "fraction2_name", "fraction3_name", "fraction4_name", "fraction5_name", "fraction6_name", "fraction7_name", "fraction8_name", "fraction9_name", "fraction10_name",
                 "raw_curve", "original_curve",
                 "peak1_percent", "peak2_percent", "peak3_percent", "peak4_percent",
                 "peak1_name", "peak2_name", "peak3_name", "peak4_name",
                 "peaks_raw_coords")
if (length(cols_keep) != length(cols_rename)) {
  stop("The number of columns to keep does not match the number of new column names!")
}

# FILTER COLUMNS
filt_dataset <- raw_dataset[ , cols_keep]
# RENAME COLUMNS
colnames(filt_dataset) <- cols_rename

# KEEP ONLY SPEP (J) and IT (F)
filt_dataset <- filt_dataset[filt_dataset$program %in% c("J", "F"), ]
print(paste0("Found data for ", nrow(filt_dataset), " rows after excluding non-J/F"))
print(paste0("Found data for ", sum(filt_dataset$program == "J"), " rows with program==J"))
print(paste0("Found data for ", sum(filt_dataset$program == "F"), " rows with program==F"))

# FOR ANALYSIS ID, WE FIRST HAVE TO SEPARATE THE ACTUAL ANALYSIS ID FROM THE TYPE OF TRACE IDENTIFIER
analysis_id_split_df <- as.data.frame(str_split_fixed(filt_dataset$analysis_id, "-", 2))
colnames(analysis_id_split_df) <- c("analysis_realid", "analysis_sequence")
print(paste0("Found data for ", sum(analysis_id_split_df$analysis_sequence == "ELP"), " rows with analysis_id_split_df==ELP"))
print(paste0("Found data for ", sum(analysis_id_split_df$analysis_sequence == "IgG"), " rows with analysis_id_split_df==IgG"))
print(paste0("Found data for ", sum(analysis_id_split_df$analysis_sequence == "IgA"), " rows with analysis_id_split_df==IgA"))
print(paste0("Found data for ", sum(analysis_id_split_df$analysis_sequence == "IgM"), " rows with analysis_id_split_df==IgM"))
print(paste0("Found data for ", sum(analysis_id_split_df$analysis_sequence == "K"), " rows with analysis_id_split_df==K"))
print(paste0("Found data for ", sum(analysis_id_split_df$analysis_sequence == "L"), " rows with analysis_id_split_df==L"))
print(paste0("Found data for ", sum(analysis_id_split_df$analysis_sequence == "Ref"), " rows with analysis_id_split_df==Ref"))
print(paste0("Found data for ", sum(analysis_id_split_df$analysis_sequence == "ELP"), " rows with analysis_id_split_df==ELP"))

# PUT NEW ID AND SEQ BACK IN THE DATASET
filt_dataset$analysis_realid <- analysis_id_split_df$analysis_realid
filt_dataset$analysis_sequence <- analysis_id_split_df$analysis_sequence
# table(filt_dataset$program, filt_dataset$analysis_sequence)

# REMOVE QUALITY CONTROLS
# filt_dataset <- filt_dataset[!grepl('^QC', filt_dataset$fullname), ]
# print(paste0("Found data for ", nrow(filt_dataset), " rows after excluding QC"))

# CHECK WHICH PATIENTS WE HAVE TO REMOVE
# WE'LL KEEP ONLY PATIENTS IF AT LEAST IGG, IGA, IGM, K, L, ELP
# CHECK WHICH IDs HAVE WHAT
for (seq_type in c("", "ELP", "IgG", "IgA", "IgM", "K", "L", "Ref")) {
  seq_type_colname = seq_type
  if (seq_type == "") {
    seq_type_colname = "RAW"
  }
  seq_type_colname = paste0(seq_type_colname, "_OK")
  # OK_seq <- analysis_seq_table$analysis_realid %in% filt_dataset$analysis_realid[analysis_id_split_df$analysis_sequence == seq_type]
  # a seq is OK if we have exactly one of this sequence for this analysis ID (else could be ex a control or a sample run multiple times... hard to understand)
  tmpp <- table(filt_dataset$analysis_realid, filt_dataset$analysis_sequence == seq_type)
  analysis_realids_OK_for_this_seq <- unique(rownames(tmpp)[tmpp[,"TRUE"] == 1])
  print(paste0("Found OK (exactly 1 trace) for ", length(analysis_realids_OK_for_this_seq), " unique analysis IDs for sequence=", seq_type))
  filt_dataset[,seq_type_colname] <- (filt_dataset$analysis_realid %in% analysis_realids_OK_for_this_seq)
}
# DECIDE WHICH TO KEEP
filt_dataset$KEEP_OK <- (filt_dataset$ELP_OK & filt_dataset$IgA_OK & filt_dataset$IgG_OK & filt_dataset$IgM_OK & filt_dataset$K_OK & filt_dataset$L_OK)

# NOW FILTER OUT ANALYSIS IDs WITHOUT COMPLETE SEQUENCE DATA
# CREATE A COPY ON WHICH WE WILL APPLY ANONYMIZATION AND DEIDENTIFICATION
compseq_dataset <- filt_dataset[filt_dataset$KEEP_OK, ]
print(paste0("Found OK data for ", nrow(compseq_dataset), " rows"))
print(paste0("Found OK data for ", length(unique(compseq_dataset$analysis_realid)), " unique analysis IDs"))

# compseq_dataset <- compseq_dataset[compseq_dataset$fullname != "\\N", ]
# print(paste0("Found data for ", nrow(compseq_dataset), " rows after excluding rows with missing patient name"))
# compseq_dataset <- compseq_dataset[compseq_dataset$birth_date != "\\N", ]
# print(paste0("Found data for ", nrow(compseq_dataset), " rows after excluding rows with missing birth date"))
# compseq_dataset <- compseq_dataset[compseq_dataset$sampling_date != "\\N", ]
# print(paste0("Found data for ", nrow(compseq_dataset), " rows after excluding rows with missing sampling date"))

# CREATE A UNIQUE ID PER PATIENT
# REPLACE \\N BY "" SO THAT WE CAN CHECK "VOID" IDs AND REMOVE THEM
compseq_dataset$patient_unique_id <- ifelse((compseq_dataset$fullname == "\\N") | (compseq_dataset$sex == "\\N") | (compseq_dataset$birth_date == "\\N"),
                                            NA,
                                            paste0(ifelse(compseq_dataset$fullname == "\\N", "[missing]", compseq_dataset$fullname),
                                                   ' (',
                                                   ifelse(compseq_dataset$sex == "\\N", "[missing]", compseq_dataset$sex),
                                                   ') ',
                                                   ifelse(compseq_dataset$birth_date == "\\N", "[missing]", substr(compseq_dataset$birth_date, 1, 10))))

aaid_paid_table <- compseq_dataset %>%
  group_by(analysis_realid) %>%
  summarise(n_unique_patient_ids = n_distinct(patient_unique_id, na.rm = TRUE)) %>%
  as.data.frame()
# table(aaid_paid_table$n_unique_patient_ids)

# analysis_realids_to_keep <- aaid_paid_table$analysis_realid[aaid_paid_table$n_unique_patient_ids > 1]
# vtmp <- compseq_dataset[compseq_dataset$analysis_realid %in% analysis_realids_to_keep, ]

# KEEP ONLY IF AN ANALYSIS ID MATCHES EXACTLY 1 UNIQUE PATIENT ID
analysis_realids_to_keep_by_paid <- aaid_paid_table$analysis_realid[aaid_paid_table$n_unique_patient_ids == 1]
compseq_dataset <- compseq_dataset[compseq_dataset$analysis_realid %in% analysis_realids_to_keep_by_paid, ]

# FILL MISSING PAID
# CREATE PAID-AAID match table
compseq_dataset_aaid_paid_table <- compseq_dataset[ ,c("analysis_realid", "patient_unique_id")]
# remove if no patient id
compseq_dataset_aaid_paid_table <- compseq_dataset_aaid_paid_table[!is.na(compseq_dataset_aaid_paid_table$patient_unique_id), ]
# then remove duplicates
compseq_dataset_aaid_paid_table <- compseq_dataset_aaid_paid_table[!duplicated(compseq_dataset_aaid_paid_table$analysis_realid), ]
# MERGE BACK -> ALL ROWS WILL HAVE AN AFFILIATED PATIENT ID
preanon_dataset <- merge(compseq_dataset[,-which(colnames(compseq_dataset) == "patient_unique_id")],
                         compseq_dataset_aaid_paid_table,
                         by = "analysis_realid", how="left")

# CREATE UNIQUE IDENTIFIER FOR EACH PATIENT
patient_id_list <- unique(preanon_dataset$patient_unique_id)
patient_id_match_table <- data.frame(patient_unique_id=patient_id_list,
                                     paid=sapply(patient_id_list, digest, algo="md5"))
patient_id_match_table$paid <- paste("PA", patient_id_match_table$paid, sep="_")
if (any(duplicated(patient_id_match_table$paid))) {
  stop("Some patient unique identifiers are not unique!")
}

# CREATE UNIQUE IDENTIFIER FOR EACH ANALYSIS
# USE REAL ID AND NOT INITIAL ID!
analysis_id_list <- unique(preanon_dataset$analysis_realid)
analysis_id_match_table <- data.frame(analysis_realid=analysis_id_list,
                                      aaid=sapply(analysis_id_list, digest, algo="md5"))
analysis_id_match_table$aaid <- paste("AN", analysis_id_match_table$aaid, sep="_")
if (any(duplicated(analysis_id_match_table$aaid))) {
  stop("Some analysis unique identifiers are not unique!")
}

# IN ANONYMIZED DATASET, COMPUTE AGE AT ANALYSIS RATHER THAN KEEPING EXACT DATES
preanon_dataset$birth_date_numeric <- as.Date(preanon_dataset$birth_date, format="%Y-%m-%d %H:%M:%S")
preanon_dataset$sampling_date_numeric <- as.Date(preanon_dataset$sampling_date, format="%Y-%m-%d %H:%M:%S")
preanon_dataset$analysis_date_numeric <- as.Date(preanon_dataset$analysis_date, format="%Y-%m-%d %H:%M:%S")

preanon_dataset$age_sampling <- as.numeric(difftime(preanon_dataset$sampling_date_numeric,
                                                    preanon_dataset$birth_date_numeric,
                                                    unit="days")) / 365.25

preanon_dataset$age_analysis <- as.numeric(difftime(preanon_dataset$analysis_date_numeric,
                                                    preanon_dataset$birth_date_numeric,
                                                    unit="days")) / 365.25

# MERGE ALL DATA (IDENTIFIED AND DEIDENTIFIED)
merged_dataset <- merge(preanon_dataset, patient_id_match_table,
                        by = "patient_unique_id", how="left")
merged_dataset <- merge(merged_dataset, analysis_id_match_table,
                        by = "analysis_realid", how="left")
if (nrow(merged_dataset) != nrow(preanon_dataset)) {
  stop("Matching failed!")
}
print(paste0("Found data for ", nrow(merged_dataset), " rows in final deidentified dataset"))
print(paste0("Found data for ", length(unique(merged_dataset$analysis_realid)), " unique analysis IDs"))
print(paste0("Found data for ", length(unique(merged_dataset$patient_unique_id)), " unique patient IDs"))

# REARRANGE COLUMNS
col_rarg <- c("paid", "patient_unique_id", "aaid", "analysis_id", "analysis_realid", "analysis_sequence",
              "fullname", "birth_date", "birth_date_numeric",
              "analysis_date", "analysis_date_numeric", "age_analysis",
              "sampling_date", "sampling_date_numeric", "age_sampling" ,         
              "program",
              "total_protein", "sex",
              "uf_code", "comment_1", "comment_long",
              "number_fractions", "fraction1_percent", "fraction2_percent",
              "fraction3_percent", "fraction4_percent", "fraction5_percent",
              "fraction6_percent", "fraction7_percent", "fraction8_percent",
              "fraction9_percent", "fraction10_percent",
              "fraction1_name", "fraction2_name", 
              "fraction3_name", "fraction4_name", "fraction5_name",
              "fraction6_name", "fraction7_name", "fraction8_name",
              "fraction9_name", "fraction10_name",
              "raw_curve", "original_curve",
              "peak1_percent", "peak2_percent", "peak3_percent",
              "peak4_percent", "peak1_name", "peak2_name",
              "peak3_name", "peak4_name", "peaks_raw_coords")
col_discard <- c("RAW_OK", "ELP_OK", "IgG_OK", "IgA_OK", "IgM_OK", "K_OK", "L_OK", "Ref_OK", "KEEP_OK")
if (!all(col_rarg %in% colnames(merged_dataset))) {
  stop("Some columns are unknown")
}
if (!all(colnames(merged_dataset) %in% c(col_rarg, col_discard))) {
  stop(paste("Some columns will be discarded without explicit mention:", paste(colnames(merged_dataset)[!(colnames(merged_dataset) %in% col_rarg)], collapse=", ")))
}

# RERRANGE COLUMNS
merged_dataset <- merged_dataset[ , col_rarg]
# SORT PATIENTS
merged_dataset <- merged_dataset[order(merged_dataset$patient_unique_id, merged_dataset$analysis_id), ]

# FINALLY CREATE TWO DATASET OUT OF THIS: THE MATCHING TABLE AND THE DEIDENTIFIED DATASET
cols_MATCH <- c("paid", "patient_unique_id", "aaid", "analysis_id", "analysis_realid", "analysis_sequence")
cols_DEIDENTIFIED <- c("paid", "aaid", "analysis_sequence",
                "age_analysis",
                "age_sampling" ,         
                "program",
                "total_protein", "sex",
                "comment_1", "comment_long",
                "number_fractions", "fraction1_percent", "fraction2_percent",
                "fraction3_percent", "fraction4_percent", "fraction5_percent",
                "fraction6_percent", "fraction7_percent", "fraction8_percent",
                "fraction9_percent", "fraction10_percent",
                "fraction1_name", "fraction2_name", 
                "fraction3_name", "fraction4_name", "fraction5_name",
                "fraction6_name", "fraction7_name", "fraction8_name",
                "fraction9_name", "fraction10_name",
                "raw_curve", "original_curve",
                "peak1_percent", "peak2_percent", "peak3_percent",
                "peak4_percent", "peak1_name", "peak2_name",
                "peak3_name", "peak4_name", "peaks_raw_coords")

# CREATE VERSIONS FOR EXPORT
EXPORT_MATCH_TABLE <- merged_dataset[, cols_MATCH]
EXPORT_DEIDENTIFIED_DATASET <- merged_dataset[, cols_DEIDENTIFIED]

# SAVE AS CSV
write.csv(EXPORT_MATCH_TABLE,
          file.path(working_directory, "EXPORT_MATCH_TABLE.csv"), row.names=FALSE)
# SAME FOR THE FULL DATASET (DEIDENTIFIED)
write.csv(EXPORT_DEIDENTIFIED_DATASET,
          file.path(working_directory, "EXPORT_DEIDENTIFIED_DATASET.csv"), row.names=FALSE)

# SAVE AS HDF (DATA ONLY) // WARNING: THIS MAY BE LONG AND HEAVY!
# h5createFile(file.path(working_directory, "EXPORT_DEIDENTIFIED_DATASET.h5"))
# h5write(EXPORT_DEIDENTIFIED_DATASET, file.path(working_directory, "EXPORT_DEIDENTIFIED_DATASET.h5"), "dataset")
# h5closeAll()

# TRY TO RELOAD H5 FILE, TO TEST IF OK
# h5f = H5Fopen(file.path(working_directory, "EXPORT_DEIDENTIFIED_DATASET.h5"))
# h5f
