($segment_file, $pdf_file, $num_file, $pos, $part_term, $offset) = @ARGV;

%id_Hash;
%utt_Hash;

if ($pos eq "")
{
	$pos = 1;
}

open(IN, $segment_file);

if ($offset eq "") { $id = 0; } else { $id = $offset; }

while ($line = <IN>)
{
	chomp $line;
	@chunk = split(/\s+/, $line);
	$name = $chunk[0];
	$id_name = $chunk[$pos];
	if ($part_term ne "")
	{
		@chunk = split(/$part_term/, $id_name);
		$id_name = $chunk[0];
	}
	$utt_Hash{$name} = $id_name;
	if ($id_Hash{$id_name} eq "")
	{
		$id_Hash{$id_name} = $id;
		$id++;
	}
}

close IN;

open(OUT, ">$num_file");

print OUT "$id\n";

close OUT;

open(IN, $pdf_file);

while ($line = <IN>)
{
	chomp $line;
	@chunk = split(/\s+/, $line);
	$name = shift @chunk;
	$len = @chunk;
	$id = $id_Hash{$utt_Hash{$name}};
	print "$name";
	for ($i=0;$i<$len;$i++)
	{
		print " $id";
	}
	print "\n";
}

close IN;
