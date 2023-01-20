#!/usr/bin/perl

# find the label from a label file according to the scp file 
# e.g. perl ../scripts/find_pdf.pl train_n1_clean_ali_stat.pdf feats_cv_train_n1_noise.scp > cv_n1_clean_ali_stat.pdf

($scpFile, $name_id_ori, $refer_id) = @ARGV;

%pdfHash;

while($line = <STDIN>)
{
	chomp($line);
    $line =~ s/^\s+//g;
    $line =~ s/\s+$//g;
    @chunk = split(/\s+/, $line);
    $name = shift @chunk;
    $str=join(" ",@chunk);
	$pdfHash{$name}=$str;
}

open(IN, $scpFile);

while($line = <IN>)
{
	chomp($line);
    $line =~ s/^\s+//g;
    $line =~ s/\s+$//g;
    @chunk = split(/\s+/, $line);
    if ($refer_id ne "true") {$name = $chunk[0];} else {$name = $chunk[1];}
    $str = $pdfHash{$name};
    if ($name_id_ori ne "true") { print $chunk[1] . " " . $str . "\n"; } else { print $chunk[0] . " " . $str . "\n"; }
}

close IN;