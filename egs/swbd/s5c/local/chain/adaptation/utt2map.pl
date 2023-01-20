($map_file, $utt_file, $nbest, $pos) = @ARGV;

%utt_Hash;

if ($pos eq "")
{
	$pos = 1;
}

open(IN, $map_file);

while ($line = <IN>)
{
	chomp $line;
	@chunk = split(/\s+/, $line);
	$name = $chunk[0];
	$map_name = $chunk[$pos];
	$utt_Hash{$name} = $map_name;
}

close IN;

open(IN, $utt_file);

while ($line = <IN>)
{
	chomp $line;
	@chunk = split(/\s+/, $line);
	$name = shift @chunk;
    $name_ori = $name;
    if ($nbest ne "false") {$name_ori =~ s/\-[^\-]+$//g;}
	$len = @chunk;
	$map_name = $utt_Hash{$name_ori};
	print "$name";
	for ($i=0;$i<$len;$i++)
	{
		print " $map_name";
	}
	print "\n";
}

close IN;


