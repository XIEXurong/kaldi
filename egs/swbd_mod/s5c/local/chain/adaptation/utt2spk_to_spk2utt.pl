#!/usr/bin/perl

%spk2utt;

while ($line = <STDIN>)
{
	chomp($line);
    $line =~ s/^\s+//g;
    $line =~ s/\s+$//g;
	@chunk = split(/\s+/, $line);
    $utt = $chunk[0];
    $spk = $chunk[1];
	$spk2utt{$spk} .= " $utt";
}

foreach $spk (sort keys %spk2utt)
{
    $utts = $spk2utt{$spk};
    print $spk . $utts . "\n";
}
