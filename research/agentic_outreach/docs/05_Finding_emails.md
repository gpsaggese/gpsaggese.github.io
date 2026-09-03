# Finding People Emails

## Hunter.Io
- Does not allow for emails from LinkedIn, but allows you to find emails
  manually with full name and company
  - <img
    src="ai_outreach_process_figs/image50.png"
    style="width:4.63611in;height:1.06244in" />

- Manual Testing with Luciana Lixandru from Sequoia Capital (~15 seconds)
  - <img
    src="ai_outreach_process_figs/image19.png"
    style="width:3.25521in;height:1.98755in" />
  - Found a different email than Phantom Buster which is weird

- Allows for bulk email finding

- Looks like easy API for email finder which can be found here:
  [https://hunter.io/api-documentation/v2#email-finder](https://hunter.io/api-documentation/v2#email-finder)

- API Testing:
  - <img
    src="ai_outreach_process_figs/image24.png"
    style="width:4.98438in;height:1.53365in" />
  - Not sure if the email is accurate
  - Very fast (~5-10 seconds)

<img
src="ai_outreach_process_figs/image16.png"
style="width:6.5in;height:2.93056in" />

<img
src="ai_outreach_process_figs/image1.png"
style="width:3.90138in;height:2.12575in" />

<img
src="ai_outreach_process_figs/image20.png"
style="width:3.23499in;height:2.24479in" />

[https://hunter.io/bulk-finders/new](https://hunter.io/bulk-finders/new)

[https://hunter.io/api-keys](https://hunter.io/api-keys)

[https://hunter.io/api-documentation/v2#email-finder](https://hunter.io/api-documentation/v2#email-finder)

## Dropcontact
- Requires file upload with any number of fields such as First Name, Last Name,
  Company, LinkedIn, and more

- Manual Testing with Luciana Lixandru profile(~2 min for 1 email)
  - Uploaded XLSX file with first name, last name, and company
  - Bulk is possible
  - Took ~2 minutes for results profile to be processed
  - Drop contact filled rest of fields such as email, phone, linkedin profile
    - <img
      src="ai_outreach_process_figs/image29.png"
      style="width:6.5in;height:0.22222in" />
  - Same email result as PhantomBuster, different result from Hunter.io

- Decent option as it can be used in conjunction with automation tools listed
  here: [https://app.dropcontact.com/api](https://app.dropcontact.com/api)

- API Testing (~15-30 seconds)
  - Posting the search query API
    - <img
      src="ai_outreach_process_figs/image55.png"
      style="width:4.87113in;height:1.97499in" />
  - Getting the results
    - <img
      src="ai_outreach_process_figs/image37.png"
      style="width:4.53646in;height:1.57758in" />
  - Comes with lots of information just with 3 easy fields, linkedin is not even
    required

<img
src="ai_outreach_process_figs/image6.png"
style="width:4.65476in;height:2.03646in" />

<img
src="ai_outreach_process_figs/image12.png"
style="width:3.38021in;height:0.8613in" />

dropcontact seems half the price for 5k emails ($.016/email)

## Snov.Io
- Very similar to Hunter.io where you use the first name, last name, and domain
  fields

- Social URL search available but not for a free account

- Testing(~1 min)
  - <img
    src="ai_outreach_process_figs/image38.png"
    style="width:5.05729in;height:2.68263in" />
  - Found same result as Hunter.io, but different from both PhantomBuster and
    DropContact results

- API
  - Not free to access
  - <img
    src="ai_outreach_process_figs/image25.png"
    style="width:3.31771in;height:3.34429in" />

- Pricing
  - <img
    src="ai_outreach_process_figs/image27.png"
    style="width:4.97396in;height:4.67903in" />
  - $0.03/ email request
  - Seems cheaper than other competitors, but questionable accuracy and long
    wait times

## Proxycurl
- Testing
  - Two types of email requests, Personal email and Work email
    - Personal email API picked up an email, but seems like it's not in use
      anymore
      - <img
        src="ai_outreach_process_figs/image5.png"
        style="width:4.57813in;height:1.3135in" />
    - Work email API did not return anything
      - <img
        src="ai_outreach_process_figs/image8.png"
        style="width:4.89063in;height:1.21174in" />
      - <img
        src="ai_outreach_process_figs/image39.png"
        style="width:3.71354in;height:1.0236in" />

- Overall, returning inaccurate emails and expensive (between $0.036 and $0.3
  depending on the plan)
