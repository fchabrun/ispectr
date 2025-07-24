# clean environment
rm(list = ls())

#########################
### REQUIRED PACKAGES ###
#########################

# load required packages
library(data.table) # to quickly load .csv files
library(ggplot2) # to make plots

############################
### ROUTINES DECLARATION ###
############################

# below we write our function which will convert the raw_curve field (i.e. hexadecimal characters)
# into structured data which may be used to train/test models
decipherIF <- function(data_line, threshold_error = 4000, threshold_low = 10000, threshold_high = 30000) {
  # arguments:
  # threshold_error (default:4000) -> used for checking errors in curves (see below)
  # threshold_low (default:10000) -> used for checking errors & for detecting markers in the curve (see below)
  # threshold_high (default:30000) -> used for detecting markers in the curve (see below)
  
  warning_flag <- 0 # declare our warning flag (useful for checking everything went ok during this routine)
  
  # first : get the hexadecimal data into a variable named s
  s <- data_line$raw_curve
  # now we break this text block into subblocks of 4 characters
  # this is because curve data is encoded by blocks of 4 hexadecimal characters
  # i.e. values between 0x0000 and 0xFFFF
  s <- gsub("(.{4})", "\\1 ", s) # add a space between each group of 4 characters
  s <- strsplit(s, " ")[[1]] # convert from str to array of str, by breaking at each blank space position
  # now we have in 's' an array of strings, each string is 4-character long
  # we use the function below in order to convert each 4-character string to its decimal equivalent
  s <- strtoi(s, 16L)
  # now, 's' contains raw values of the curve
  # we cannot stop here, because the curves themselves contain remarkable points
  # all values above 30k are actually markers
  # for immunotypings, those markers are only found at the beginning and the end of the curve, i.e. first and last points
  # so we'll have to post-process the curve

  # first, we look for missing data -> if any, we'll replace those by 0's and provide a warning flag when outputing the results
  if (sum(is.na(s)) > 0) {
    warning_flag <- bitwOr(warning_flag, 1) # add 1 to the warning flag
    s[is.na(s)] <- 0 # replace NA values by 0's
  }
  
  # we reverse the curve, because we're used to interpreting curves with albumin first and gamma last
  s <- rev(s)

  # we perform a quick quality check
  # some curves have abnormalities : those curves start or end
  # by a repetition of high values (around 5-15 values in a row close to 4k)
  # while normally, all curves should start and end by values close to 0
  # in order to highlight those curves,
  # we compute the derivative of the curve (diff() function, or setdiff1d for python's numpy)
  # so we can check the deltas, pixel wise
  # and we parse the curve in order to find "jumps" from 0-ish values to 4k-ish values
  # beware : we remind that the curve contains 'marker's, i.e. values higher than 30k
  # allowing to find remarkable points
  # so we don't want to detect jumps from regular values in the curves (i.e. between 0 and 4k) to those values (i.e. around 30k)
  # thus, we'll only detect jumps higher than 4k but lower than 10k
  
  d <- diff(s)
  if (sum(abs(d) > threshold_error & abs(d) < threshold_low) > 0) {
    # if we're here : we have detected an abnormality
    warning_flag <- bitwOr(warning_flag, 2) # we put the 2 flag 'on'
    # we won't correct the error
    # because the origin of those errors is unknown
    # so we don't want to risk keeping abnormal curves
  }
  
  # now, we'll find those markers we previously mentioned
  # i.e. values higher than 30k
  # there also are some markers with values around 10k (>10k)
  # but it seems that we usually only find those markers on SPEs, not on immunotyping curves
  
  # first, we list the positions on the curve where there is a value higher than 10k
  delimitations <- which(s >= threshold_low)
  
  # now, we get the exact values at those positions
  delimitations_values <- s[delimitations] # on extrait la valeur correspondant à chacune de ces délimitations, pour pouvoir séparer les normales des secondaires (sous-fractions)
  
  # below is a little bit more complicated
  # we are going to remove those values from the final curve
  # however, we want to keep trace of were those markers were
  # but since we'll remove them, the initial positions we determined are going to change
  # e.g : if we have a curve starting with : 1, 2, 32k, 3, 32k, 4 -> the markers are at positions 3 and 5 of the curve
  # however, after removing thoses values, the new curve will be : 1, 2, 3, 4 -> the marker should be point at positions 2 (after point 2) and 3 (after point 3)
  # so below we subtract our positions in order to update their final positions on the output curve
  delimitations <- delimitations - 1:length(delimitations)
  
  # finally, we separate those delimitations into 2 classes
  # fractions, including start and end markers
  # and subfractions, i.e. markers with values higher than 10k but lower than 30k
  # as written above, those markers are not expected in immunotyping curves
  fractions <- delimitations[delimitations_values >= threshold_high]
  subfractions <- delimitations[delimitations_values < threshold_high]
  
  # finally, we remove all those markers from the final curve
  values <- s[s < threshold_low]

  # and we return our output :
  return(list(curve = values, # curve values
              fractions_names = rep('?',length(fractions)-1), # fractions names -> do not use
              fractions_coords = fractions, # fractions markers -> for ITs, only start and end markers
              subfractions_coords = numeric(0), # subfractions markers -> discard, since they're not expected and useless
              peaks_coords = numeric(0), # m-spikes coordinates -> discard, since usually ITs should not contain m-spikes data
              alertflag = warning_flag)) # our warning flag
}

# this function takes a data.frame with 7 rows corresponding to each curve obtained from one sample
decipherWholeIF <- function(data_lines) {
  # create the empty data.frame
  df=data.frame(x=rep(seq(1:300),7),
                y=numeric(300*7),
                m=character(300*7),
                stringsAsFactors = FALSE)
  # for each data_line in data_lines -> i.e. each curve obtained by immunotyping (G, A, M...)
  for(i in 1:nrow(data_lines)) {
    # convert the hexadecimal data to a structured curve
    new_curve = decipherIF(data_lines[i,])$curve
    # add that curve to the df
    df[((i-1)*300+1):((i-1)*300+length(new_curve)),"y"]=new_curve
    df[((i-1)*300+1):((i-1)*300+length(new_curve)),"m"]=data_lines[i,]$method
  }
  df=df[nchar(df$m)>0,] # remove empty lines
  # return output
  return(df)
}

############################
### PIPELINE ###
############################

### LOAD DATA

# replace by path to Appendix file
path_to_edited_sql_file = "./CSV_data_1.csv"

raw_data <- as.data.frame(fread(path_to_edited_sql_file, header = T, sep = ',', quote=""))

raw_data <- raw_data[,c("id", "curva")] # keep only id and curve
# the id contains both the actual id of the sample (e.g. 00000001111) and the method used (igg, iga, igm...)
# we need to parse ids to get this info
raw_data$method <- gsub("^[^-]+-", "", raw_data$id) # get method (IgG, IgA, IgM, K, L or Ref)
raw_data$aid <- gsub("-.+$", "", raw_data$id) # get "analysis id" (aid)
# re order columns
raw_data <- raw_data[,c("aid","method","curva")]
colnames(raw_data) <- c("aid","method","raw_curve")

# list samples (i.e. unique 'aid')
analysis_ids <- unique(raw_data$aid)

### EXTRACT STRUCTURED DATA, SAMPLE-WISE

# define which data we want to keep, and in which order : first reference (=ELP), then IgG, A, M and finally kappa and lambda curves
array_order = c("ELP","IgG","IgA","IgM","K","L")
# declare empty array for extracting structure data
# by default we choose 298-point curves
data_array = array(NA, dim=c(length(analysis_ids),length(array_order),298))

# parse analyses

# do not mind the lines below, they are used for calling a custom written library
# in order to check the progress of our loop (equivalent to python's tqdm)
# you may simply comment this line
# library(pbar)
# my_pb <- pb$new(length(analysis_ids))

for(i in 1:length(analysis_ids)) { # for each sample unique ID
  # get the sample ID at position i
  chosen_ID = analysis_ids[i]
  # get all data rows corresponding to this sample
  data_lines <- raw_data[raw_data$aid==chosen_ID,]
  
  if (all(array_order %in% data_lines$method)) { # all methods listed should be in this subset : ELP, IgG, A, M, kappa and lambda curves
    
    tmp=decipherWholeIF(data_lines) # use our custom function to convert those data to structured data (see above)
    
    # check all curves are here and same length
    # we just check that we have each curve among the ones selected in the variable "array_order"
    # and we check that all curves are 298 points long
    prev_l = 298
    OK = T
    for(arr in array_order) {
      if(sum(tmp$m==arr) != prev_l) {
        OK = F
        break
      }
    }
    if (OK) { # we only continue past here if we have all curves and all are the same length
      # fill the array
      for(arr in 1:length(array_order)) {
        # we get the raw curve
        unpadded_curve <- tmp[tmp$m==array_order[arr],]$y
        # we store the curve in our array
        data_array[i,arr,] = unpadded_curve
      }
    }
  }
  # my_pb$update() # this function updates our progress bar, it should be commented as well if the declaration of "my_pb" was commented
  
  # beware : this loop is pretty slow
  # it is expected to achieve around 150-160 items per second
  # e.g. 15 seconds for ~2.5k samples here
  # a better alternative should be found for large datasets
}

sum(is.na(data_array)) # expected: 0 -> everything went well ! :)

# now we will convert our array into a data.frame which may be exported as a .csv file
final_df <- data.frame(AID = analysis_ids)

for(i in 1:length(array_order)) {
  tmp_df <- as.data.frame(data_array[,i,])
  colnames(tmp_df) <- paste0(array_order[i], 'x', 1:298)
  final_df <- cbind(final_df, tmp_df)
}

dim(final_df) # the final dataset should have 1789 columns (298 points x 6 dimensions = 1788 + 1 for the analysis id)

# replace by path to Appendix file (output)
path_to_output_file = "./CSV_data_2.csv"

# export for python
write.csv(x = final_df,
          file = path_to_output_file,
          row.names = F,
          quote = F)
