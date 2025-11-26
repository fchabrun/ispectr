rm(list = ls())

library(RJSONIO)
library(ggplot2)
library(ggpubr)
library(extrafont)

json_dir = "C:/Users/flori/OneDrive - univ-angers.fr/Documents/Home/Research/SPECTR/ISPECTR/data/2025/lemans/preannotation/output_jsons"
output_dir = "C:/Users/flori/OneDrive - univ-angers.fr/Documents/Home/Research/SPECTR/ISPECTR/output/temp"
json_filenames = list.files(json_dir)

# json_filename = json_filenames[j]
json_filename = "AN_009b1a3166da7572b24fd3e9f4c5c701.json"

json_data = fromJSON(file.path(json_dir, paste0(json_filename)))

traces = list()
trace_names = c("ELP", "IgG", "IgA", "IgM", "K", "L")
trace_names_full = c("A. Original trace", "B. After gamma chains precipitation", "C. After alpha chains precipitation",
                     "D. After mu chains precipitation", "E. After kappa chains precipitation", "F. After lambda chains precipitation")
for (i in 1:length(trace_names)) {
  trace_name <- trace_names[i]
  
  gp <- ggplot() +
    geom_line(data=data.frame(x=1:304, y=json_data$traces[[trace_name]]$data), mapping=aes(x=x, y=y),
              size=1) +
    ylab("Relative abundance (%)") +
    xlab("Time (s)") +
    scale_y_continuous(labels = scales::percent) +
    ggtitle(trace_names_full[i]) +
    theme_minimal() +
    theme(plot.title=element_text(size=14, family="Arial"))
  
  traces[[trace_name]] <- gp
}

png(file.path(output_dir, paste0(substr(json_filename, 1, nchar(json_filename) - 4), "png")),
    res=300,
    width=3000, height=2400)
gp <- ggarrange(traces[["ELP"]], traces[["IgG"]],
          traces[["IgA"]], traces[["IgM"]],
          traces[["K"]], traces[["L"]],
          # labels = trace_names_full,
          ncol = 2, nrow = 3)
print(gp)
dev.off()

