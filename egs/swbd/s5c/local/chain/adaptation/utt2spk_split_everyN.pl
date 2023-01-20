($utt2spk_list, $N) = @ARGV;

use POSIX

%utt2spk_hash;
%spk2utt_hash;
$utt_all = "";
$spk_all = "";

open (IN, $utt2spk_list);

while ($line = <IN>)
{
	$line =~ s/^\s+//g;
	chomp $line;
	
	@chunk = split(/\s+/, $line);
	$utt_name = $chunk[0];
	$spk = $chunk[1];
	
	$spk2utt_hash{$spk} .= " $utt_name";
	$utt_all .= " $utt_name";
	$spk_all .= " $spk";
}

close IN;

$spk_all =~ s/^\s+//g;
@spk_all_arr = split(/\s+/, $spk_all);

foreach $spk (@spk_all_arr)
{
	$utt_all_spk = $spk2utt_hash{$spk};
	$utt_all_spk =~ s/^\s+//g;
	@utt_all_spk_arr = split(/\s+/, $utt_all_spk);
	
	$utt_num = @utt_all_spk_arr;
	$utt_num_sub = $N;
	
	for ($i=1; $i<=$utt_num; $i++)
	{
		$j = POSIX::ceil($i/$utt_num_sub);
		$utt = $utt_all_spk_arr[$i-1];
		$utt2spk_hash{$utt} = $spk . "_sub" . $j;
	}
}

$utt_all =~ s/^\s+//g;
@utt_all_arr = split(/\s+/, $utt_all);

foreach $utt (@utt_all_arr)
{
	$spk = $utt2spk_hash{$utt};
	print "$utt $spk\n";
}
